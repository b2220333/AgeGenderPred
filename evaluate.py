import os
import re
import cv2
import glob

from config import config
from train import AgePredModel
from preprocess import clear_dir


def evaluate(path,
             name_contain_label = False,
             result_path = None):
  """
  function used to test a folder of images
  :param path: folder path, e.g. config.val, last char should be '/'
  :param name_contain_label: if set, label is extracted from image name
  :param result_path: path to store the result, default path + "test_results/"
  :return: 
  """
  # check param: path
  if path[-1] != "/":
    print("[WARNING] PARAM: path NOT ENDS WITH '/'!")
    path += '/'

  # check param: result_path
  if result_path is None:
    result_path = path + "test_results/"

  # make sure it exists and is empty
  if not os.path.exists(result_path):
    os.mkdir(result_path)
  clear_dir(result_path)

  # start eval
  model = AgePredModel(eval_use_only=True)
  int2gender = {0: 'Female', 1: 'Male'}
  for img_path in glob.glob(path + "*"):
    # check image
    img_name = img_path[len(path):]
    formatt = re.findall("[^.]*.([^.]*)", img_name)[0]
    if not formatt: continue
    formatt = formatt.lower()
    if not formatt in ['png', 'jpg', 'jpeg']: continue
    print("[evaluate] Eval {} ...".format(img_name), end='')

    # input image
    preds, rects, scores = model.getAgeGender(img_path,
                                              transformed=False,
                                              return_all_faces=True,
                                              return_info=True)

    img = cv2.imread(img_path)

    for (gender_pred, age_pred), (x, y, w, h), score in zip(preds, rects, scores):
      age_pred, gender_pred = float(age_pred), int(gender_pred)

      # draw a rectange to bound the face
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)

      # fill an area with color for text show
      color = (255, 0, 0) if gender_pred else (0, 0, 255)
      cv2.rectangle(img, (x, y + h - 35), (x + w, y + h), color, cv2.FILLED)

      # put text
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(img,
                  "{}, {:.0f}, {:.2f}".format(int2gender[gender_pred], age_pred, score),
                  org=(x + 6, y + h - 6),
                  fontFace=font,
                  fontScale=1.0,
                  color=(255, 255, 255),
                  thickness=1)

      # bottomLeftCornerOfText = (min(img.shape[1], x + 4), max(0, y - 12))
      # if x < max(24., 0.05 * img.shape[0]):
      #   bottomLeftCornerOfText = (min(img.shape[1], x + 4), min(img.shape[1], y + h - 2))
      # fontScale = 1
      # fontColor = (255, 0, 0) if gender_pred else (0, 0, 255)
      # lineType = 2
      # cv2.putText(img,
      #             "{}, {:.0f}, {:.2f}".format(int2gender[gender_pred], age_pred, score),
      #             bottomLeftCornerOfText,
      #             font,
      #             fontScale,
      #             fontColor,
      #             lineType)
    cv2.imwrite(result_path + img_name, img)
    print(" Done!")


if __name__ == "__main__":
  evaluate(config.pics + "self/",
           name_contain_label=False)




















