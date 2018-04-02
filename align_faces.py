import cv2
import dlib
import imutils
from imutils.face_utils import FaceAligner as FA
from imutils.face_utils import rect_to_bb

from config import config

class FaceAligner:

  def __init__(self):
    self.resizeWidth      = 600 # input resize
    self.desiredFaceWidth = 256 # output size
    self.Path2ShapePred   = config.model + "shape_predictor_68_face_landmarks.dat"

    self.detector         = dlib.get_frontal_face_detector()
    self.predictor        = dlib.shape_predictor(self.Path2ShapePred)
    self.fa               = FA(self.predictor, desiredFaceWidth=self.desiredFaceWidth)

  def getAligns(self,
                img,
                savepath = None,
                return_rects = False,
                use_RBG = False):
    """
    get face alignment picture
    :param img: original BGR image or a path to it
    :param savepath: savepath, format "xx/xx/xx.png"
    :param return_rects: if set, return face positinos [(x, y, w, h)] 
    :return: aligned faces, (opt) rects
    """
    if type(img) == str:
      img = cv2.imread(img)
    elif use_RBG:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = self.detector(gray, 2)
    aligned = [self.fa.align(img, gray, rect) for rect in rects]

    if use_RBG:
      aligned = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in aligned]

    if savepath:
      if len(aligned) == 1:
        cv2.imwrite(savepath, aligned)
      else:
        for i, al in enumerate(aligned):
          cv2.imwrite("{}_{}.{}".format(savepath[:-4], i, savepath[-3:]), aligned)

    if return_rects:
      return aligned, [rect_to_bb(rect) for rect in rects]
    return aligned # BGR faces, cv2.imshow("Aligned", faceAligned)

  def test(self):
    image = cv2.imread("images/example_02.jpg")
    # image = imutils.resize(image, width=self.resizeWidth)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale image
    cv2.imshow("Input", image)
    rects = self.detector(gray, 2)

    # loop over the face detections
    for rect in rects:
      # extract the ROI of the *original* face, then align the face
      # using facial landmarks
      (x, y, w, h) = rect_to_bb(rect)
      faceOrig = imutils.resize(image[y:y + h, x:x + w], width=self.desiredFaceWidth)
      faceAligned = self.fa.align(image, gray, rect)

      import uuid
      f = str(uuid.uuid4())
      cv2.imwrite("foo/" + f + ".png", faceAligned)

      # display the output images
      cv2.imshow("Original", faceOrig)
      cv2.imshow("Aligned", faceAligned)
      cv2.waitKey(0)


if __name__ == "__main__":
  ts = FaceAligner()
  ts.test()
  pass








