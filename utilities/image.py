import cv2
import logging

def readImage(imagePath):
  logging.info('Reading image from -> {}'.format(imagePath))
  return cv2.imread(imagePath)

def writeImage(image, path):
  logging.info('Writing image to -> {}'.format(path))
  cv2.imwrite(path, image)

