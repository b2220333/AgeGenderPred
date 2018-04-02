import os
import time

def time_count(fn):
  # Funtion wrapper used to memsure time consumption
  def _wrapper(*args, **kwargs):
    start = time.clock()
    result = fn(*args, **kwargs)
    print(">>[Time Count]: Funtion '%s' Costs %fs" % (fn.__name__, time.clock() - start))
    return result
  return _wrapper

def CreatPathIfNotExists(fn):
  def _wrapper(*args, **kwargs):
    result = fn(*args, **kwargs)
    if not os.path.exists(result):
      os.makedirs(result)
    return result
  return _wrapper



class Config:

  def __init__(self):
    self.root = os.getcwd() + '/'

  @property
  @CreatPathIfNotExists
  def model(self):
    return self.root + "models/"

  @property
  @CreatPathIfNotExists
  def pics(self):
    return self.root + "pics/"

  @property
  @CreatPathIfNotExists
  def wiki_raw(self):
    return self.pics + "wiki_crop/"

  @property
  @CreatPathIfNotExists
  def wiki_labeled(self):
    return self.pics + "wiki_labeled/"

  @property
  @CreatPathIfNotExists
  def imdb_raw(self):
    return self.pics + "imdb_crop/"

  @property
  @CreatPathIfNotExists
  def imdb_labeled(self):
    return self.pics + "imdb_labeled/"

  @property
  @CreatPathIfNotExists
  def aligned(self):
    return self.pics + "aligned/"

  @property
  @CreatPathIfNotExists
  def train(self):
    return self.pics + "train/"

  @property
  @CreatPathIfNotExists
  def val(self):
    return self.pics + "val/"





config = Config()




















