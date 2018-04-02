import os
import cv2
import time
import random
import shutil
import glob
import scipy.io

import numpy as np

from shutil import copyfile
from datetime import datetime, timedelta
from multiprocessing import Pool

from config import config
from align_faces import FaceAligner



# -*- coding: utf-8 -*-

def convertMatlabDate(x):
  x, date = int(x), -1
  try:
    date = (datetime.fromordinal(int(x))
          + timedelta(days=x % 1)
          - timedelta(days=366)).year
  except:
    print("[Error] On Processing {}".format(x))
  return date


def clear_dir(path):
  if os.path.exists(path):
    shutil.rmtree(path)
    os.mkdir(path)


def addlabels(data = 'wiki'):
  # 1, clean previous
  origin_dir = config.wiki_raw if data == 'wiki' else config.imdb_raw
  obj_dir = config.wiki_labeled if data == 'wiki' else config.imdb_labeled
  clear_dir(obj_dir)
  # 2, read meta data
  mat = scipy.io.loadmat(origin_dir + data + '.mat')[data][0][0]
  for dob, dop, path, gender, name in zip(mat[0][0], mat[1][0], mat[2][0], mat[3][0], mat[4][0]):
    age = dop - convertMatlabDate(dob)
    # 2.0 drop unqualified pics
    if age < 0 or age > 120: continue
    if gender not in [1.0, 0.0]: continue
    newName = "{}_{}_{}.jpg".format(age,
                                    int(gender),
                                    name[0]
                                    .replace(' ', '')
                                    .replace('/', '')
                                    .replace(':', ''))
    # 2.1 check duplicate
    newNameNoDupli = newName
    while os.path.exists(obj_dir + newNameNoDupli):
      newNameNoDupli = "{}{}{}".format(newName[:-4], random.randint(1, 10000), newName[-4:])
    # 2.2 save as a new file
    copyfile(origin_dir + path[0], obj_dir + newNameNoDupli)


# public
FL = FaceAligner()
def sub_align_face(picname):
  aligned = FL.getAligns(picname)
  if len(aligned) != 1: return
  cv2.imwrite(config.aligned + picname, aligned[0])

def align_faces(data = 'wiki', clean = False):
  origin_dir = config.wiki_labeled if data == 'wiki' else config.imdb_labeled
  if clean: clear_dir(config.aligned)
  os.chdir(origin_dir)
  jobs = glob.glob("*.jpg")
  with Pool() as pool:
    try:
      pool.map(sub_align_face, jobs)
    finally:
      pool.close()


def sub_divideTrainVal(img):
  if np.random.rand() < 0.95:
    copyfile(config.aligned + img, config.train + img)
  else:
    copyfile(config.aligned + img, config.val + img)

def divideTrainVal():
  # os
  pwt = os.getcwd()
  os.chdir(config.aligned)

  # read into mem
  train, val = [], []
  for img in glob.glob("*.jpg"):
    if np.random.rand() < 0.95:
      train.append([cv2.imread(img), img])
    else:
      val.append([cv2.imread(img), img])

  # dump out of mem
  for img, name in train:
    cv2.imwrite(config.train + img, name)
  for img, name in val:
    cv2.imwrite(config.val + img, name)
  os.chdir(pwt)


if __name__ == "__main__":
  # addlabels(data='wiki')
  # align_faces(data='wiki')
  divideTrainVal()













