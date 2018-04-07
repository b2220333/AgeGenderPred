import os
import re
import glob
import torch
import numpy as np

from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset

from config import parser
parser = parser['DATA']


class FaceDataset(Dataset):
  """ read images from disk dynamically """

  def __init__(self, datapath, transformer):
    """
    init function
    :param datapath: datapath to aligned folder  
    :param transformer: image transformer
    """
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
    """
    get images and labels
    :param idx: image index 
    :return: image: transformed image, gender: torch.LongTensor, age: torch.FloatTensor
    """
    img_name = self.datapath + self.pics[idx]
    image = transforms.ToPILImage()(io.imread(img_name))
    (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", self.pics[idx])[0]
    gender = torch.from_numpy(np.array([gender], dtype='float')).type(torch.LongTensor)
    age = torch.from_numpy(np.array([float(age) / float(parser['age_divide'])], dtype='float')).type(torch.FloatTensor)
    if self.transformer:
      image = self.transformer(image)
    else:
      image = torch.from_numpy(image)
    return image, gender, age



class FaceDatasetInMem(Dataset):
  """ 
  accelorate the loding process... only use this when u have 
  enough memory! Process is amost the same as FaceDataset except
  loadIntoMem() load images into mem first.
  """

  def __init__(self, datapath, transformer):
    """
    init function
    :param datapath: datapath to aligned folder  
    :param transformer: image transformer
    """
    if datapath[-1] != '/':
      print("[WARNING] PARAM: datapath SHOULD END WITH '/'")
      datapath += '/'
    self.datapath     = datapath
    self.pics         = [f[len(datapath) : ] for f in
                         glob.glob(datapath + "*.jpg")]
    self.transformer  = transformer
    self.loadIntoMem()

  def loadIntoMem(self):
    """
    load data into memory for fast loading
    :return: 
    """
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
      age = torch.from_numpy(np.array([float(age) / float(parser['age_divide'])], dtype='float')).type(torch.FloatTensor)
      self.labels.append([gender, age])

  def __len__(self):
    return len(self.pics)

  def __getitem__(self, idx):
    """
    get images and labels
    :param idx: image index 
    :return: image: transformed image, gender: torch.LongTensor, age: torch.FloatTensor
    """
    image = self.imgs[idx]
    (gender, age) = self.labels[idx]
    return image, gender, age