import logging
from detectors.detector import Detector
from detectors.detectorlist import detectorList as detectors

class AllDetectors(Detector):

  def __init__(self):
    logging.debug('initializing all detectors')
    self.detectors = [className() for (className, _, _) in detectors]

  def detectFacesInImage(self, img):
    faces = []
    for detector in self.detectors:
      detectedFaces = detector.detectFacesInImage(img)
      [faces.append(face) for face in detectedFaces]
    
    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return faces
