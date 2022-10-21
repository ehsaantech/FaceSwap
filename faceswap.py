import logging
import cv2
from face_swap import mask_from_points, warp_image_3d
import numpy
from utilities.cli import takeCliOptionSelection
from utilities.image import readImage, writeImage
from utilities.videoprocessor import VideoProcessor
from images.imagelist import imageList as images
from videos.videolist import videoList as videos
from detectors.detectorlist import detectorList as simpleDetectors
from detectors.aggregatorslist import detectorList as aggregatedDetectors
from predictors.predictorlist import predictorList as predictors

def faceswapCli(outputDirName) -> str:
  outputDirPath = '{0}/{1}'.format(outputDirName, 'faceswap')
  try:
    detectors = simpleDetectors + aggregatedDetectors
    detectorSelection = takeCliOptionSelection([d[1] for d in detectors], 'detector')
    predictorSelection = takeCliOptionSelection([p[1] for p in predictors], 'predictor')
    sourceSelection = takeCliOptionSelection([img[0] for img in images], 'source image')

    imagesCopy = images.copy()
    imagesCopy.pop(sourceSelection)
    targetOptions = imagesCopy + videos
    targetSelection = takeCliOptionSelection([t[0] for t in targetOptions], 'target')

    (detectorClass, detectorName, _) = detectors[detectorSelection]
    (predictorClass, predictorName) = predictors[predictorSelection]
    (srcPath, srcName, _) = images[sourceSelection]
    (targetPath, targetName, targetExtension) = targetOptions[targetSelection]

    detector = detectorClass()
    predictor = predictorClass()

    outputPath = '{0}/{1}-{2}-{3}-{4}.{5}'.format(outputDirPath, srcName, targetName, predictorName, detectorName, targetExtension)

    isVideo = targetExtension == 'mp4'
    if isVideo:
      faceSwapInVideo(detector, predictor, srcPath, targetPath, outputPath)
    else:
      faceSwap(detector, predictor, srcPath, targetPath, outputPath)

  except Exception as exception:
    logging.warning(exception)
    return

  return outputDirPath

def faceSwap(detector, predictor, srcPath, targetPath, outputPath):
  srcImage = readImage(srcPath)
  srcFaces = detector.detectFacesInImage(srcImage)
  if len(srcFaces) != 1:
    logging.info('source image should have 1 face. Aborting!')
    return
  srcFaceFeatures = predictor.predictFeaturesInImage(srcImage, srcFaces[0])

  targetImage = readImage(targetPath)
  targetFaces = detector.detectFacesInImage(targetImage)
  if len(targetFaces) == 0:
      return writeImage(targetImage, outputPath)

  targetFacesWithFeatures = [predictor.predictFeaturesInImage(targetImage, face) for face in targetFaces]

  outputImage = targetImage
  for targetFaceFeatures in targetFacesWithFeatures:
    outputImage = _faceSwap(srcImage, srcFaceFeatures, outputImage, targetFaceFeatures)
  writeImage(outputImage, outputPath)

def faceSwapInVideo(detector, predictor, srcPath, targetPath, outputPath):
  srcImage = readImage(srcPath)
  srcFaces = detector.detectFacesInImage(srcImage)
  if len(srcFaces) != 1:
    logging.info('source image should have 1 face. Aborting!')
    return
  srcFaceFeatures = predictor.predictFeaturesInImage(srcImage, srcFaces[0])
  
  def process(image):
    faces = detector.detectFacesInImage(image)
    if len(faces) == 0:
      return image

    facesWithFeatures = [predictor.predictFeaturesInImage(image, face) for face in faces]
    outputImage = image
    for targetFaceFeatures in facesWithFeatures:
      outputImage = _faceSwap(srcImage, srcFaceFeatures, outputImage, targetFaceFeatures)
    return outputImage
  
  videoprocessor = VideoProcessor(targetPath, outputPath, process)
  videoprocessor.start()

def _faceSwap(srcImage, srcFaceFeatures, targetImage, targetFacesWithFeatures):
  outputImage = face_swap_custom(srcImage, targetImage, srcFaceFeatures, targetFacesWithFeatures)
  return outputImage

def face_swap_custom(srcImage, targetImage, srcFaceFeatures, targetFaceFeatures):
  h, w = targetImage.shape[:2]
  warpedImage = warp_image_3d(srcImage, srcFaceFeatures, targetFaceFeatures, (h, w))

  # Mask for blending
  mask = mask_from_points((h, w), targetFaceFeatures, erode_flag=False)
  mask_src = numpy.mean(warpedImage, axis=2) > 0
  mask = numpy.asarray(mask * mask_src, dtype=numpy.uint8)

  #Poisson Blending
  r = cv2.boundingRect(mask)
  center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
  output = cv2.seamlessClone(warpedImage, targetImage, mask, center, cv2.NORMAL_CLONE)
  return output
    