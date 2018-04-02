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

from config import config
from align_faces import FaceAligner
from model import AgePredModel
from preprocess import clear_dir

def evaluate(img_path = config.val):
  clear_dir(img_path + 'test_result')
  if not os.path.exists(img_path + 'test_result'):
    os.mkdir(img_path + 'test_result')
  model = AgePredModel()
  int2gender = {0: 'Female', 1: 'Male'}
  for img in glob.glob(img_path + "*.jpg"):
    img_name = img[len(img_path): ]
    (age_true, gender_true) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", img_name)[0]
    (gender_pred, age_pred) = model.getAgeGender(img,
                                                 aligned=False,
                                                 transformed=False)
    age_true, gender_true = float(age_true), int(gender_true)
    age_pred, gender_pred = float(age_pred), int(gender_pred)
    fig, ax = plt.subplots(1, 1)
    plt.axis('off')
    ax.imshow(io.imread(img))
    ax.set_title("Age P = {:.1f} | T = {:.1f} || Gender P = {} | T = {}"
                 .format(age_pred,
                         age_true,
                         int2gender[gender_pred],
                         int2gender[gender_true]))
    plt.savefig(img_path + 'test_result/' + img_name)
    plt.clf()


if __name__ == "__main__":
  evaluate()




















