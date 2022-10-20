import dlib
import logging
from detectors.detector import Detector
from detectors.detectorProperties import detectionConfidence

class MModDetector(Detector):

  def __init__(self):
    logging.debug('initializing mmod_human_face_detector')
    self.detector = dlib.cnn_face_detection_model_v1('detectors/models/mmod_human_face_detector.dat')
  
  def detectFacesInImage(self, img):
    detections = self.detector(img, upsample_num_times = 1)
    faces = []
    for detection in detections:
      confidence = detection.confidence
      if confidence > detectionConfidence:
        faces.append(dlib.rectangle(detection.rect))
    
    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return faces
