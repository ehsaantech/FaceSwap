import dlib
import logging
from detectors.detector import Detector

class HogSvmFaceDetector(Detector):

  def __init__(self):
    logging.debug('initializing hog_svm_face_detector')
    self.detector = dlib.get_frontal_face_detector()

  def detectFacesInImage(self, img):
    faces = self.detector(img, upsample_num_times = 1)
    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return [ dlib.rectangle(face) for face in faces ]
