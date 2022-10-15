import dlib
import logging
from detectors.detector import Detector

class MModDetector(Detector):

  def __init__(self):
    logging.debug('initializing mmod_human_face_detector')
    self.detector = dlib.cnn_face_detection_model_v1('detectors/mmod_human_face_detector.dat')
  
  def detectFacesInImage(self, img):
    faces = self.detector(img, upsample_num_times = 1)
    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return [ dlib.rectangle(face.rect) for face in faces ]
