import logging
import cv2
import dlib
import mediapipe
from detectors.detector import Detector
from detectors.detectorProperties import detectionConfidence

mp_face_detection = mediapipe.solutions.face_detection
# mp_drawing = mediapipe.solutions.drawing_utils

class MediapipeDetector(Detector):

  def __init__(self):
    logging.debug('initializing mediapipe_detector')
  
  def detectFacesInImage(self, img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
      result = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

      if not result.detections:
        return []


    faces = []
    for face in result.detections:
      confidence = face.score[0]
      if confidence > detectionConfidence:
        boundingBox = face.location_data.relative_bounding_box
        (imgHeight, imgWidth, _) = img.shape
          
        x = int(boundingBox.xmin * imgWidth)
        w = int(boundingBox.width * imgWidth)
        y = int(boundingBox.ymin * imgHeight)
        h = int(boundingBox.height * imgHeight)

        rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
        faces.append(rect)

    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return faces

  # def markFacesInImage(self, image, faces):
  #   imageCopy = image.copy()
  #   for detection in faces:
  #     mp_drawing.draw_detection(imageCopy, detection)

  #   return imageCopy