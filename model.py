import os
import re
import cv2
import time
import copy
import glob
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io, transform
from shutil import copyfile
from math import sqrt
from multiprocessing import cpu_count

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable


from config import config
from align_faces import FaceAligner

class FaceDataset(Dataset):
  """ read images from disk dynamically """

  def __init__(self, datapath, transformer):
    if datapath[-1] != '/':
      print("[WARNING] PARAM: datapath SHOULD END WITH '/'")
      datapath += '/'
    self.datapath     = datapath
    self.pics         = [f[len(datapath) : ] for f in
                         glob.glob(datapath + "*.jpg")]
    self.transformer  = transformer

  def __len__(self):
    return len(self.pics)

  def __getitem__(self, idx):
    img_name = self.datapath + self.pics[idx]
    image = transforms.ToPILImage()(io.imread(img_name))
    (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", self.pics[idx])[0]
    gender = torch.from_numpy(np.array([gender], dtype='float')).type(torch.LongTensor)
    age = torch.from_numpy(np.array([float(age) / 10.], dtype='float')).type(torch.FloatTensor)
    if self.transformer:
      image = self.transformer(image)
    else:
      image = torch.from_numpy(image)
    return image, gender, age


class FaceDatasetInMem(Dataset):
  """ accelorate the loding process... only use this when u have 
      enough memory !"""

  def __init__(self, datapath, transformer):
    if datapath[-1] != '/':
      print("[WARNING] PARAM: datapath SHOULD END WITH '/'")
      datapath += '/'
    self.datapath     = datapath
    self.pics         = [f[len(datapath) : ] for f in
                         glob.glob(datapath + "*.jpg")]
    self.transformer  = transformer
    self.loadIntoMem()

  def loadIntoMem(self):
    self.imgs, self.labels = [], []
    for name in self.pics:
      # add image
      path = os.path.join(self.datapath, name)
      img = transforms.ToPILImage()(io.imread(path))
      if self.transformer:
        img = self.transformer(img)
      else:
        img = torch.from_numpy(img)
      self.imgs.append(img)

      # add label
      (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", name)[0]
      gender = torch.from_numpy(np.array([gender], dtype='float')).type(torch.LongTensor)
      age = torch.from_numpy(np.array([float(age) / 10.], dtype='float')).type(torch.FloatTensor)
      self.labels.append([gender, age])

  def __len__(self):
    return len(self.pics)

  def __getitem__(self, idx):
    image = self.imgs[idx]
    (gender, age) = self.labels[idx]
    return image, gender, age

"""
RestNet-18 Archetecture
0 Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
1 BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
2 ReLU(inplace)
3 MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
4 Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  )
  (1): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  )
)
5 Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (downsample): Sequential(
      (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (1): BasicBlock(
    (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  )
)
6 Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (downsample): Sequential(
      (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (1): BasicBlock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
  )
)
7 Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (downsample): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (1): BasicBlock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
  )
)
8 AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
9 Linear(in_features=512, out_features=1000, bias=True)

"""
class NNetwork(torch.nn.Module):
  def __init__(self, onlyTrainLastLayers):
    super(NNetwork, self).__init__()
    self.resNet = models.resnet18(pretrained=True)

    # use pretrained model
    if onlyTrainLastLayers:
      for i, child in enumerate(self.resNet.children()):
        if i >= 6: break
        for param in child.parameters():
          param.requires_grad = False

    # set up residual FCN
    in_feat   = self.resNet.fc.in_features
    out_feat  = 256
    self.resNet.fc = nn.Linear(in_feat, out_feat)

    self.ageLinear1 = nn.Linear(out_feat, out_feat)
    self.ageLinear2 = nn.Linear(out_feat, out_feat)
    self.ageOut     = nn.Linear(out_feat, 1)

    self.genLinear1 = nn.Linear(out_feat, out_feat)
    self.genLinear2 = nn.Linear(out_feat, out_feat)
    self.genOut     = nn.Linear(out_feat, 2)

  def forward(self, x):
    residule = self.resNet(x)

    age_pred = F.relu(self.ageLinear1(residule))
    age_pred = F.relu(self.ageLinear2(age_pred))
    age_pred += residule
    age_pred = F.relu(self.ageOut(age_pred))

    gender_pred = F.relu(self.genLinear1(residule))
    gender_pred = F.relu(self.genLinear2(gender_pred))
    gender_pred += residule
    gender_pred = F.log_softmax(self.genOut(gender_pred), dim=1)

    return gender_pred, age_pred


class AgePredModel:

  def __init__(self,
               onlyTrainLastLayers = True,
               train_from_scratch = False):
    """
    age prediction model, provide APIs for public uses
    :param onlyTrainLastLayers: is set, freeze params of previous layers
    :param train_from_scratch: 
    """
    self.from_scratch     = train_from_scratch
    self.lr_rate          = 1e-3
    self.batch_size       = 256
    self.model            = NNetwork(onlyTrainLastLayers)
    self.use_gpu          = torch.cuda.is_available()
    self.checkpoint_best  = config.model + "best_torch.nn"
    self.latest_weights   = config.model + "latest_torch.nn"
    self.csv_path         = config.model + "log.csv"

    self.age_criterion    = nn.L1Loss()
    self.gender_criterion = nn.NLLLoss()

    self.loaded = False
    self.aligner = FaceAligner()

    if self.use_gpu:
      self.model = self.model.cuda()
      self.age_criterion = self.age_criterion.cuda()
      self.gender_criterion = self.gender_criterion.cuda()

    self.transformer = {
      'train': transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }

    columns = ['Timstamp', 'Epoch', 'Phase', 'Gender Acc', 'Age MAE', 'Best Gender Acc', 'Best Age MAE', 'Lr_rate']
    if not self.from_scratch and os.path.exists(self.csv_path):
      self.csv_checkpoint = pd.read_csv(self.csv_path)
    else:
      self.csv_checkpoint  = pd.DataFrame(data=[], columns=columns)

    self.load_data()
    # pass

  @staticmethod
  def img2matrix():
    pwt = os.getcwd()
    os.chdir(config.aligned)
    data, labels = [], []
    for name in glob.glob("*.jpg"):
      (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", name)[0]
      lbl = (float(gender), float(age))
      img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
      data.append(img), labels.append(lbl)
    data, labels = np.array(data), np.array(labels)
    np.save(config.pics + 'imgs.npy', data)
    np.save(config.pics + 'labels.npy', labels)
    os.chdir(pwt)

  def load_data(self):
    print("[AgePredModel] load_data: start loading...")
    image_datasets = {x: FaceDataset(config.pics + x + '/',
                                     self.transformer[x])
                      for x in ['train', 'val']}
    self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=min(16, cpu_count()))
                        for x in ['train', 'val']}
    self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("[AgePredModel] load_data: Done! Get {} for train and {} for test!".format(self.dataset_sizes['train'],
                                                                   self.dataset_sizes['val']))
    print("[AgePredModel] load_data: loading finished !")


  # TODO: Double Check the Training LOOP !!!
  def train_model(self,
                  num_epochs=32):
    print(self.model)
    print("[AgePredModel] train_model: Start training...")
    since = time.time()

    best_model_wts = copy.deepcopy(self.model.state_dict())
    best_gender_acc, best_age_loss = .0, 100
    didnt_reduce_rounds = 0

    # init optimizer
    self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=self.lr_rate,
                                weight_decay=1e-4)

    # load weights if possible
    if not self.from_scratch and os.path.exists(self.checkpoint_best):
      try:
        checkpoint = torch.load(self.checkpoint_best)
        self.model.load_state_dict(checkpoint['state_dic'])
        best_gender_acc = checkpoint['best_gender_acc']
        best_age_loss   = checkpoint['best_age_loss']
        best_model_wts  = checkpoint['state_dic']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("[train_model] Load Weights Successful")
      except:
        pass

    # start each epoch
    for epoch in range(num_epochs):
      print('Start Epoch {}/{} ...'.format(epoch, num_epochs - 1))
      print('-' * 10)

      # Each epoch has a training and validation phase
      for phase in ['train', 'val']:
        # shift train/eval model
        self.model.train(phase == 'train')

        epoch_age_loss = 0.0
        epoch_gender_tp = 0
        processed_data = 0

        # Iterate over data.
        epoch_start_time = time.time()
        for data in self.dataloaders[phase]:
          # get the inputs
          inputs, gender_true, age_true = data
          processed_data += self.batch_size

          # wrap inputs/oputpus in Variable
          if self.use_gpu:
            inputs = Variable(inputs.cuda())
            gender_true = Variable(gender_true.cuda())
            age_true = Variable(age_true.cuda())
          else:
            inputs = Variable(inputs)
            gender_true = Variable(gender_true)
            age_true = Variable(age_true)
          # inputs.volatile = True

          # zero gradients
          self.optimizer.zero_grad()

          # forward and get output
          gender_out, age_pred = self.model(inputs)
          _, gender_pred = torch.max(gender_out, 1)
          gender_true = gender_true.view(-1)

          # get loss
          gender_loss = self.gender_criterion(gender_out, gender_true)
          age_loss = self.age_criterion(age_pred, age_true)
          loss = age_loss + gender_loss

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            self.optimizer.step()

          # statistics
          gender_pred = gender_pred.cpu().data.numpy()
          gender_true = gender_true.cpu().data.numpy()
          this_epoch_gender_tp = np.sum(gender_pred == gender_true)
          epoch_age_loss += age_loss.data[0] * inputs.size(0)
          epoch_gender_tp += this_epoch_gender_tp

          # print info after a batch done
          print("|| {:.2f}% {}/{} || Gender Loss {:.4f} || Age MAE {:.2f} || Acc {:.2f}% |"
                "| LR {} || ETA {:.0f}s || BEST AGE {:.2f} || BEST GENDER {:.2f}% ||"
                .format(100 * processed_data / self.dataset_sizes[phase],
                        processed_data,
                        self.dataset_sizes[phase],
                        gender_loss.cpu().data.numpy()[0],
                        10 * age_loss.cpu().data.numpy()[0],
                        100 * this_epoch_gender_tp / self.batch_size,
                        self.lr_rate,
                        (self.dataset_sizes[phase] - processed_data) * ( time.time() - epoch_start_time) / processed_data,
                        10 * best_age_loss,
                        100 * best_gender_acc))

        # epoch done
        epoch_gender_acc = epoch_gender_tp / self.dataset_sizes[phase]
        epoch_age_loss /= self.dataset_sizes[phase]

        # print info after epoch done
        print('\n\n{} {}/{} Done! \t\t Age MAE = {:.2f} \t\t Gender Acc {:.2f}% Lr = {:.5f}\n\n'
              .format(phase.upper(),
                      epoch,
                      num_epochs,
                      10 * epoch_age_loss,
                      100 * epoch_gender_acc,
                      self.lr_rate))

        # deep copy the model
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

        # save csv file
        self.csv_checkpoint.loc[len(self.csv_checkpoint)] = [str(datetime.datetime.now()),
                                                             epoch,
                                                             phase,
                                                             epoch_gender_acc,
                                                             10 * epoch_age_loss,
                                                             best_gender_acc,
                                                             10 * best_age_loss,
                                                             self.lr_rate]
        self.csv_checkpoint.to_csv(config.model + "log.csv", index=False)

        # reduce learning rate by 1/10, minimum lr = 1e-5
        if phase == "train" and didnt_reduce_rounds >= 4 and self.lr_rate > 1e-5:
          self.lr_rate /= 2
          print("\n[didnt_reduce_rounds] Reduce Learning Rate From {} --> {} \n".format(self.lr_rate * 2, self.lr_rate))
          for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_rate
          didnt_reduce_rounds = 0

        # empty catch to free gpu mem
        torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
    print('Best val enderACC: {:.4f} AgeMAE: {:.0f}'.format(best_gender_acc, torch.sqrt(best_age_loss)))

    # load best model weights
    self.model.load_state_dict(best_model_wts)
    return self.model


  def getAgeGender(self,
                   img,
                   aligned = False,
                   transformed = False):
    if not self.loaded:
      checkpoint = torch.load(self.checkpoint_best, map_location='cpu')
      self.model.load_state_dict(checkpoint['state_dic'])
      self.model.train(False)
      self.loaded = True

    if type(img) == str:
      img = io.imread(img)

    # TODO: now we only detect 1 face by default
    # TODO: change: if no face detected, use original image
    if not aligned:
      aligned = self.aligner.getAligns(img, use_RBG=True)
      if len(aligned): img = aligned[0]

    img = transforms.ToPILImage()(img)

    if not transformed:
      img = self.transformer['val'](img)

    img = Variable(torch.unsqueeze(img, 0))
    gender_out, age_pred = self.model(img)
    _, gender_pred = torch.max(gender_out, 1)
    gender_pred = gender_pred.cpu().data.numpy()[0]
    age_pred    = 10 * age_pred.cpu().data.numpy()[0][0]

    return gender_pred, age_pred



if __name__ == "__main__":
  a = AgePredModel()
  # a.train_model(32)
  print(a.getAgeGender(config.val + "6_0_MurderofElisaIzquierdo.jpg"))
  # a.divideTrainVal()
  # a.img2matrix()
  # face_dataset = FaceDataset()
  # print(face_dataset[1])









