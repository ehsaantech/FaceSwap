from abc import ABC, abstractmethod
import logging
import random
import cv2

class Predictor(ABC):

  @abstractmethod
  def predictFeaturesInImage(self, img, faceRect):
    pass

  def markFeaturesInImage(self, image, facesWithFeatures):
    imageCopy = image.copy()
    for faceFeatures in facesWithFeatures:
      logging.debug('Marking face on image {}'.format(faceFeatures))
      color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
      for featurePoint in faceFeatures:
        cv2.circle(imageCopy, featurePoint, 2, color, thickness=2)
    
    return imageCopy

  def markFaceHullsInImage(self, image, facesWithFeatures):
    hullsForFaces = []
    for faceFeatures in facesWithFeatures:
      logging.debug('Getting convex hull of face on image {}'.format(faceFeatures))
  
      hullPoints = []
      hullIndices = cv2.convexHull(faceFeatures, returnPoints = False)

      for i in range(0, len(hullIndices)):
        hullPoints.append(faceFeatures[hullIndices[i, 0]])
  
      logging.debug('Marking convex hull of face on image {}'.format(hullPoints))
      hullsForFaces.append(hullPoints)

    return self.markFeaturesInImage(image, hullsForFaces)
