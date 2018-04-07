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

from config import config, parser
from align_faces import FaceAligner

parser = parser['DATA']

def convertMatlabDate(x):
  """
  conver date format in IMDB-Wiki dataset, which is 
   matlab date format, to python int format represent years
  :param x: matlab date
  :return: int, year
  """
  x, date = int(x), -1
  try:
    date = (datetime.fromordinal(int(x))
          + timedelta(days=x % 1)
          - timedelta(days=366)).year
  except:
    print("[Error] On Processing {}".format(x))
  return date


def clear_dir(path):
  """
  clean all files in this directionary
  :param path: path to dir
  :return: 
  """
  if os.path.exists(path):
    shutil.rmtree(path)
    os.mkdir(path)
  return


def addlabels(data = 'wiki'):
  """
  move pictures to labled dir and rename to Age_Gender_Name.jpg format
  :param data: 'wiki' or 'imdb'
  :return: 
  """
  # 1, clean previous
  origin_dir = config.wiki_raw if data == 'wiki' else config.imdb_raw
  obj_dir = config.wiki_labeled if data == 'wiki' else config.imdb_labeled
  clear_dir(obj_dir)

  # 2, read meta data
  mat = scipy.io.loadmat(origin_dir + data + '.mat')[data][0][0]
  for dob, dop, path, gender, name in zip(mat[0][0], mat[1][0], mat[2][0], mat[3][0], mat[4][0]):
    age = dop - convertMatlabDate(dob)
    # 2.0 drop unqualified pics
    if age < int(parser['age_lower']) or age > int(parser['age_upper']): continue
    if gender not in [1.0, 0.0]: continue
    newName = "{}_{}_{}.jpg".format(age,
                                    int(gender),
                                    name[0]
                                    .replace(' ', '')
                                    .replace('/', '')
                                    .replace(':', ''))
    # 2.1 check duplicate
    # 2.1 if duplicate exist, append a random number to it name
    newNameNoDupli = newName
    while os.path.exists(obj_dir + newNameNoDupli):
      newNameNoDupli = "{}{}{}".format(newName[:-4], random.randint(1, 9999), newName[-4:])
    # 2.2 save as a new file
    copyfile(origin_dir + path[0], obj_dir + newNameNoDupli)
  return

# TODO: any other ways to get around this public variable?
FL = FaceAligner()
def sub_align_face(picname):
  """
  sub thread function to get and store aligned faces
  :param picname: pic names
  :return: 
  """
  aligned = FL.getAligns(picname)
  if len(aligned) != 1: return
  cv2.imwrite(config.aligned + picname, aligned[0])
  return

def align_faces(data = 'wiki', clean = False):
  """
  get aligned faces from labeled folder and store it in aligned folder for training
  :param data: 'wiki' or 'imdb'
  :param clean: if set, clean aligned folder, else append or rewrite to it
  :return: 
  """
  origin_dir = config.wiki_labeled if data == 'wiki' else config.imdb_labeled
  if clean: clear_dir(config.aligned)
  os.chdir(origin_dir)
  jobs = glob.glob("*.jpg")

  # un-parallel
  # for picname in jobs:
  #   aligned = FL.getAligns(picname)
  #   if len(aligned) != 1: return
  #   cv2.imwrite(config.aligned + picname, aligned[0])

  # parallel
  with Pool() as pool:
    try:
      pool.map(sub_align_face, jobs)
    finally:
      pool.close()
  return


def sub_divideTrainVal(img):
  """
  distribute images randomly to train or test foled by 95% train prob
  :param img: image path
  :return: 
  """
  if np.random.rand() < float(parser['train_test_div']):
    copyfile(config.aligned + img, config.train + img)
  else:
    copyfile(config.aligned + img, config.val + img)
  return

def divideTrainVal():
  """
  distribute images randomly to train or test foled by 95% train prob
  :return: 
  """
  pwt = os.getcwd()
  os.chdir(config.aligned)

  # clean
  clear_dir(config.train)
  clear_dir(config.val)

  # read into mem
  # train, val = [], []

  # parallel
  with Pool() as pool:
    try:
      pool.map(sub_divideTrainVal, glob.glob("*.jpg"))
    finally:
      pool.close()

  # for img in glob.glob("*.jpg"):
  #   if np.random.rand() < float(parser['train_test_div']):
  #     cv2.imwrite(config.train + img, cv2.imread(img))
      # train.append([cv2.imread(img), img])
    # else:
    #   cv2.imwrite(config.val + img, cv2.imread(img))
      # val.append([cv2.imread(img), img])

  # dump out of mem
  # for img, name in train:
  #   cv2.imwrite(config.train + img, name)
  # for img, name in val:
  #   cv2.imwrite(config.val + img, name)
  os.chdir(pwt)
  return


if __name__ == "__main__":
  # addlabels(data='wiki')
  # print('wiki..')
  # align_faces(data='wiki', clean = True)
  # print('imdb..')
  # align_faces(data='imdb')
  print('divide...')
  divideTrainVal()
  pass













