import dlib
import logging
import cv2
from detectors.detector import Detector
from detectors.detectorProperties import detectionConfidence

class DnnCaffeDetector(Detector):

  def __init__(self):
    logging.debug('initializing caffe_dnn_model')
    modelFile = "detectors/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "detectors/models/deploy.prototxt"
    self.detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    self.detector.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

  def detectFacesInImage(self, img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123], swapRB=False, crop=False)
    self.detector.setInput(blob)
    detections = self.detector.forward()

    faces = []
    for face in detections[0][0]:
      confidence = face[2]
      if confidence > detectionConfidence:
        bbox = face[3:]

        left = int(bbox[0] * width)
        top = int(bbox[1] * height)
        right = int(bbox[2] * width)
        bottom = int(bbox[3] * height)

        faces.append(dlib.rectangle(left=left, top=top, right=right, bottom=bottom))
    
    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return faces