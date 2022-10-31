import cv2
import numpy
import logging

def readImage(imagePath):
  logging.info('Reading image from -> {}'.format(imagePath))
  return cv2.imread(imagePath)

def writeImage(image, path):
  logging.info('Writing image to -> {}'.format(path))
  cv2.imwrite(path, image)

def cropImage(image, rect, pad=10):
  height, width = image.shape[:2]
  top = min(max(0, rect.top() - pad), height)
  left = min(max(0, rect.left() - pad), width)
  bottom = max(min(height, rect.bottom() + pad), 0)
  right = max(min(width, rect.right() + pad), 0)
  logging.debug('cropping image with (w,h) -> ({0},{1}) to (l,t,r,b) -> ({2},{3},{4},{5})'.format(width, height, left, top, right, bottom))
  return image[top:bottom, left:right]

