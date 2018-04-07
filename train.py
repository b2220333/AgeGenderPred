import os
import re
import cv2
import time
import copy
import glob
import datetime
import numpy as np
import pandas as pd

from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable

from config import config, parser
from retrained_resnet_model import RetrainedResnetModel, resnet_transformer
from align_faces import FaceAligner
from dataload import FaceDataset


class AgePredModel:
  """ train/test class for age/gender prediction """

  def __init__(self,
               train_from_scratch = False,
               eval_use_only = False):
    """
    age prediction model, provide APIs for public uses
    :param onlyTrainLastLayers: is set, freeze params of previous layers
    :param train_from_scratch: if set, do not load pretrained params
    :param eval_use_only: if set, model will not load training/testing data
    """
    self.from_scratch     = train_from_scratch
    self.model            = RetrainedResnetModel()
    self.use_gpu          = torch.cuda.is_available()
    self.transformer      = resnet_transformer()
    self.checkpoint_best  = config.model + "best_torch.nn"
    self.checkpoint_last  = config.model + "last_torch.nn"
    self.latest_weights   = config.model + "latest_torch.nn"
    self.csv_path         = config.model + "log.csv"

    self.batch_size       = int(parser['TRAIN']['batch_size'])
    self.num_epochs       = int(parser['TRAIN']['num_epochs'])
    self.loading_jobs     = int(parser['TRAIN']['jobs_to_load_data'])
    self.max_no_reduce    = int(parser['TRAIN']['max_no_reduce'])

    self.weight_decay     = float(parser['TRAIN']['weight_decay'])
    self.age_divide       = float(parser['DATA']['age_divide'])
    self.min_lr_rate      = float(parser['TRAIN']['min_lr_rate'])
    self.lr_reduce_by     = float(parser['TRAIN']['lr_reduce_by'])
    self.lr_rate          = float(parser['TRAIN']['init_lr_rate'])

    self.loaded           = False
    self.age_criterion    = nn.L1Loss()
    self.gender_criterion = nn.NLLLoss()
    self.aligner          = FaceAligner()

    if self.use_gpu:
      self.model = self.model.cuda()
      self.age_criterion = self.age_criterion.cuda()
      self.gender_criterion = self.gender_criterion.cuda()

    columns = ['Timstamp', 'Epoch', 'Phase', 'Gender Acc',
               'Age MAE', 'Best Gender Acc', 'Best Age MAE', 'Lr_rate']
    self.csv_checkpoint = pd.DataFrame(data=[], columns=columns)
    if not self.from_scratch and os.path.exists(self.csv_path):
      self.csv_checkpoint = pd.read_csv(self.csv_path)

    if not eval_use_only:
      self.load_data()

  def load_data(self):
    print("[AgePredModel] load_data: start loading...")
    image_datasets = {x: FaceDataset(config.pics + x + '/',
                                     self.transformer[x])
                      for x in ['train', 'val']}
    self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.loading_jobs)
                        for x in ['train', 'val']}
    self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("[AgePredModel] load_data: Done! Get {} for train and {} for test!"
          .format(self.dataset_sizes['train'],
                  self.dataset_sizes['val']))
    print("[AgePredModel] load_data: loading finished !")


  # TODO: Double Check the Training LOOP !!!
  def train_model(self):
    print(self.model)
    print("[AgePredModel] train_model: Start training...")
    since = time.time()

    # 1.0.0.0 Define Perams
    best_model_wts = copy.deepcopy(self.model.state_dict())
    best_gender_acc, best_age_loss = .0, 100
    didnt_reduce_rounds = 0

    # 2.0.0.0 init optimizer
    self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=self.lr_rate,
                                weight_decay=self.weight_decay)

    # 3.0.0.0 load weights if possible
    if not self.from_scratch and os.path.exists(self.checkpoint_best):
      try:
        checkpoint = torch.load(self.checkpoint_best)
        self.model.load_state_dict(checkpoint['state_dic'], strict=False)
        best_gender_acc = checkpoint['best_gender_acc']
        best_age_loss   = checkpoint['best_age_loss']
        best_model_wts  = checkpoint['state_dic']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("[train_model] Load Weights Successful")
      except:
        pass

    # 4.0.0.0 start each epoch
    for epoch in range(self.num_epochs):
      print('Start Epoch {}/{} ...'.format(epoch, self.num_epochs - 1))
      print('-' * 10)

      # 4.1.0.0 loop over training and validation phase
      for phase in ['train', 'val']:
        # 4.1.1.0 shift train/eval model
        self.model.train(phase == 'train')
        torch.cuda.empty_cache()

        epoch_age_loss = 0.0
        epoch_gender_tp = 0
        processed_data = 0

        # 4.1.2.0 Iterate over data.
        epoch_start_time = time.time()
        for data in self.dataloaders[phase]:
          # 4.1.2.1 get the inputs and labels
          inputs, gender_true, age_true = data
          processed_data += self.batch_size

          # 4.1.2.2 wrap inputs&oputpus into Variable
          #         NOTE: set voloatile = True when
          #         doing evaluation helps reduce
          #         gpu mem usage.
          volatile = phase == 'val'
          if self.use_gpu:
            inputs = Variable(inputs.cuda(), volatile=volatile)
            gender_true = Variable(gender_true.cuda(), volatile=volatile)
            age_true = Variable(age_true.cuda(), volatile=volatile)
          else:
            inputs = Variable(inputs, volatile=volatile)
            gender_true = Variable(gender_true, volatile=volatile)
            age_true = Variable(age_true, volatile=volatile)

          # 4.1.2.3 zero gradients
          self.optimizer.zero_grad()

          # 4.1.2.4 forward and get outputs
          gender_out, age_pred = self.model(inputs)
          _, gender_pred = torch.max(gender_out, 1)
          gender_true = gender_true.view(-1)

          # 4.1.2.5 get loss
          gender_loss = self.gender_criterion(gender_out, gender_true)
          age_loss = self.age_criterion(age_pred, age_true)
          loss = age_loss + gender_loss

          # 4.1.2.6 backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            self.optimizer.step()

          # 4.1.2.7 statistics
          gender_pred = gender_pred.cpu().data.numpy()
          gender_true = gender_true.cpu().data.numpy()
          this_epoch_gender_tp = np.sum(gender_pred == gender_true)
          epoch_age_loss += age_loss.data[0] * inputs.size(0)
          epoch_gender_tp += this_epoch_gender_tp

          # 4.1.2.8 print info for each bach
          print("|| {:.2f}% {}/{} || Gender Loss {:.4f} || Age MAE {:.2f} || Acc {:.2f}% |"
                "| LR {} || ETA {:.0f}s || BEST AGE {:.2f} || BEST GENDER {:.2f}% ||"
                .format(100 * processed_data / self.dataset_sizes[phase],
                        processed_data,
                        self.dataset_sizes[phase],
                        gender_loss.cpu().data.numpy()[0],
                        self.age_divide * age_loss.cpu().data.numpy()[0],
                        100 * this_epoch_gender_tp / self.batch_size,
                        self.lr_rate,
                        (self.dataset_sizes[phase] - processed_data) * ( time.time() - epoch_start_time) / processed_data,
                        self.age_divide * best_age_loss,
                        100 * best_gender_acc))

          # 4.1.2.9 free up mem
          del inputs, gender_true, age_true
          del gender_loss, age_loss, loss

        # 4.1.3.0 epoch done
        epoch_gender_acc = epoch_gender_tp / self.dataset_sizes[phase]
        epoch_age_loss /= self.dataset_sizes[phase]

        # 4.1.4.0 print info after epoch done
        print('\n\n{} {}/{} Done! \t\t Age MAE = {:.2f} \t\t Gender Acc {:.2f}% Lr = {:.5f}\n\n'
              .format(phase.upper(),
                      epoch,
                      self.num_epochs,
                      self.age_divide * epoch_age_loss,
                      100 * epoch_gender_acc,
                      self.lr_rate))

        # 4.1.5.0, save model weights
        if phase == 'val' and epoch_age_loss < best_age_loss:
          best_gender_acc = epoch_gender_acc
          best_age_loss = epoch_age_loss
          best_model_wts = copy.deepcopy(self.model.state_dict())
          torch.save({'epoch': epoch,
                      'state_dic': best_model_wts,
                      "best_gender_acc": best_gender_acc,
                      "best_age_loss": best_age_loss,
                      "optimizer": self.optimizer.state_dict()
                      }, self.checkpoint_best)
          didnt_reduce_rounds = 0
          print("\n\tNew BEST FOUND!! Age Loss = {:.2f}, Gender Acc = {:.2f}\n"
                .format(best_age_loss, best_gender_acc))
        elif phase == 'train':
          didnt_reduce_rounds += 1
          torch.save({'epoch': epoch,
                      'state_dic': self.model.state_dict(),
                      "best_gender_acc": best_gender_acc,
                      "best_age_loss": best_age_loss,
                      "optimizer": self.optimizer.state_dict()
                      }, self.checkpoint_last)

        # 4.1.6.0 save csv logging file
        self.csv_checkpoint.loc[len(self.csv_checkpoint)] = [str(datetime.datetime.now()),
                                                             epoch,
                                                             phase,
                                                             epoch_gender_acc,
                                                             self.age_divide * epoch_age_loss,
                                                             best_gender_acc,
                                                             self.age_divide * best_age_loss,
                                                             self.lr_rate]
        self.csv_checkpoint.to_csv(config.model + "log.csv", index=False)

        # 4.1.7.0 reduce learning rate if nessessary
        if phase == "train" \
                and didnt_reduce_rounds >= self.max_no_reduce\
                and self.lr_rate > self.min_lr_rate:
          self.lr_rate /= self.lr_reduce_by
          print("\n[didnt_reduce_rounds] Reduce Learning Rate From {} --> {} \n"
                .format(self.lr_rate * self.lr_reduce_by, self.lr_rate))
          for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_rate
          didnt_reduce_rounds = 0

      # 4.2.0.0 train/val loop ends

    # 5.0.0.0 Trainning Complete!
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
    print('Best val enderACC: {:.4f} AgeMAE: {:.0f}'.format(best_gender_acc, torch.sqrt(best_age_loss)))

    # 6.0.0.0 load best model weights
    self.model.load_state_dict(best_model_wts)
    return self.model


  def getAgeGender(self,
                   img,
                   transformed = False,
                   return_all_faces = True,
                   return_info = False):
    """
    evaluation/test funtion
    :param img: str or numpy array represent the image
    :param transformed: if the image is transformed into standarlized pytorch image
            applicable when using this in train loop
    :param return_all_faces: if set, return prediction results of all faces detected,
            applicable if it's known that all images comtain only 1 face
    :param return_info: if set, return a list of [ (x, y, w, h) ] represents loc of faces
    :return: a list of [gender_pred, age_pred]
    """
    # load model params
    if not self.loaded:
      checkpoint = torch.load(self.checkpoint_best,
                              map_location='gpu' if self.use_gpu else 'cpu')
      self.model.load_state_dict(checkpoint['state_dic'])
      self.model.train(False)
      self.loaded = True

    # load images if not provided
    if type(img) == str:
      img = cv2.imread(img)
    # img = deepcopy(img)

    aligned = self.aligner.getAligns(img, return_info= return_info)

    # TODO: Check this default rects
    if return_info:
      aligned, rects, scores = aligned

    if not len(aligned):
      scores = [1]
      rects = [(0, 0, img.shape[0], img.shape[1])]
      faces = [img]
    else:
      faces = aligned

    if not return_all_faces:
      faces = faces[0]

    faces = [transforms.ToPILImage()(fc) for fc in faces]

    if not transformed:
      faces = [self.transformer['val'](fc) for fc in faces]

    preds = []
    for face in faces:
      face = Variable(torch.unsqueeze(face, 0))
      gender_out, age_pred = self.model(face)
      _, gender_pred = torch.max(gender_out, 1)
      gender_pred = gender_pred.cpu().data.numpy()[0]
      age_pred = 10 * age_pred.cpu().data.numpy()[0][0]
      preds += [(gender_pred, age_pred)]

    if return_info:
      return preds, rects, scores
    return preds

if __name__ == "__main__":
  a = AgePredModel()
  a.train_model()
  # print(a.getAgeGender(config.val + "6_0_MurderofElisaIzquierdo.jpg"))
  # a.divideTrainVal()
  # a.img2matrix()
  # face_dataset = FaceDataset()
  # print(face_dataset[1])









