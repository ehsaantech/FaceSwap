import dlib
import logging
import cv2
from detectors.detector import Detector

class HaarCascadeDetector(Detector):

  def __init__(self, name, scaleFactor, minNeighbors):
    logging.debug('initializing {}'.format(name))
    model = '{0}{1}'.format(cv2.data.haarcascades, name)
    self.detector = cv2.CascadeClassifier(model)

    self.scaleFactor = scaleFactor
    self.minNeighbors = minNeighbors

    # sebDetectorModel = '{}haarcascade_eye.xml'.format(cv2.data.haarcascades)
    # self.sebDetector = cv2.CascadeClassifier(sebDetectorModel)

  def _detect(self, img, detector):
    rects = detector.detectMultiScale(img, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, )
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

  def detectFacesInImage(self, img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.equalizeHist(grayscale)
    detections = self._detect(grayscale, self.detector)
    
    faces = []
    for (left, top, right, bottom) in detections:
      # cropped = grayscale[top:bottom, left:right]
      # subDetections = self._detect(cropped, self.sebDetector)
      # if len(subDetections) > 0:
        # faces.append(dlib.rectangle(left=left, top=top, right=right, bottom=bottom))
      faces.append(dlib.rectangle(left=left, top=top, right=right, bottom=bottom))

    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return faces


haarcascade_default = lambda : HaarCascadeDetector('haarcascade_frontalface_default.xml', scaleFactor=1.1, minNeighbors=6) 
haarcascade_alt = lambda : HaarCascadeDetector('haarcascade_frontalface_alt.xml', scaleFactor=1.1, minNeighbors=3)
haarcascade_alt2 = lambda : HaarCascadeDetector('haarcascade_frontalface_alt2.xml', scaleFactor=1.1, minNeighbors=3)
haarcascade_alt_tree = lambda : HaarCascadeDetector('haarcascade_frontalface_alt_tree.xml', scaleFactor=1.1, minNeighbors=1)