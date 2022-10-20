import logging
import dlib
import cv2
from mtcnn import MTCNN
from detectors.detector import Detector
from detectors.detectorProperties import detectionConfidence

class MtcnnDetector(Detector):

  def __init__(self):
    logging.debug('initializing mtcnn')
    self.detector = MTCNN()

  def detectFacesInImage(self, img):
    detections = self.detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    faces = []
    for detection in detections:
      confidence = float(detection['confidence'])
      if confidence > detectionConfidence:
        (left, top, width, height) = detection['box']
        right = left + width
        bottom = top + height
        faces.append(dlib.rectangle(left=left, top=top, right=right, bottom=bottom))
    
    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return faces