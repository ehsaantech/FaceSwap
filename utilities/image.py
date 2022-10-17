import cv2
import numpy
import logging

def readImage(imagePath):
  logging.info('Reading image from -> {}'.format(imagePath))
  return cv2.imread(imagePath)

def writeImage(image, path):
  logging.info('Writing image to -> {}'.format(path))
  cv2.imwrite(path, image)

def cropImage(image, points, pad=10):
  left, top = numpy.min(points, 0) - pad
  right, bottom = numpy.max(points, 0) + pad
  return image[top:bottom, left:right]

