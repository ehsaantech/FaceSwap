from abc import ABC, abstractmethod
import logging
import random
import cv2

class Detector(ABC):

  @abstractmethod
  def detectFacesInImage(self, image):
    pass

  def markFacesInImage(self, image, faces):
    imageCopy = image.copy()
    for face in faces:
      logging.debug('Marking face on image {}'.format(face))
      color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
      cv2.rectangle(imageCopy, (face.left(), face.top()), (face.right(), face.bottom()), color, thickness=2)

    return imageCopy

  def process(self, image):
    faces = self.detectFacesInImage(image)
    return self.markFacesInImage(image, faces)

