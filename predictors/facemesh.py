import logging
import cv2
import dlib
import mediapipe as mp
import numpy
from predictors.predictor import Predictor
from utilities.image import cropImage, writeImage

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceMeshPredictor(Predictor):

  def __init__(self):
    logging.debug('initializing mediapipe_facemesh')
    self.cropPadding = 0
  
  def predictFeaturesInImage(self, img, faceRect):
    croppedImage = cropImage(img, faceRect, pad=0)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7) as face_mesh:
      # Convert the BGR image to RGB before processing.
      result = face_mesh.process(cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB))

      # Print and draw face mesh landmarks on the image.
      if not result.multi_face_landmarks:
        logging.debug("found 0 shapes")
        return []

      height = croppedImage.shape[0]
      width = croppedImage.shape[1]
      top = faceRect.top()
      left = faceRect.left()
      landmarks = result.multi_face_landmarks[0].landmark
      coords = numpy.asarray([((landmark.x * width) + left, (landmark.y * height) + top) for landmark in landmarks], dtype=int)
      return coords