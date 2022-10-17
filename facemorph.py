import logging
from face_swap import face_swap_custom
from utilities.cli import takeCliOptionSelection
from utilities.image import readImage, writeImage
from utilities.videoprocessor import VideoProcessor
from images.imagelist import imageList as images
from videos.videolist import videoList as videos
from detectors.detectorlist import detectorList as detectors
from predictors.predictorlist import predictorList as predictors

def morphCli(outputDirName) -> str:
  outputDirPath = '{0}/{1}'.format(outputDirName, 'morph')
  try:
    detectorSelection = takeCliOptionSelection([d[1] for d in detectors], 'detector')
    predictorSelection = takeCliOptionSelection([p[1] for p in predictors], 'predictor')
    sourceSelection = takeCliOptionSelection([img[0] for img in images], 'source image')

    imagesCopy = images.copy()
    imagesCopy.pop(sourceSelection)
    targetOptions = imagesCopy + videos
    targetSelection = takeCliOptionSelection([t[0] for t in targetOptions], 'target')

    (detectorClass, detectorName) = detectors[detectorSelection]
    (predictorClass, predictorName) = predictors[predictorSelection]
    (srcPath, srcName, _) = images[sourceSelection]
    (targetPath, targetName, targetExtension) = targetOptions[targetSelection]

    detector = detectorClass()
    predictor = predictorClass()

    outputPath = '{0}/{1}-{2}-{3}-{4}.{5}'.format(outputDirPath, srcName, targetName, predictorName, detectorName, targetExtension)

    isVideo = targetExtension == 'mp4'
    if isVideo:
      faceMorphInVideo(detector, predictor, srcPath, targetPath, outputPath)
    else:
      faceMorph(detector, predictor, srcPath, targetPath, outputPath)

  except Exception as exception:
    logging.warning(exception)
    return

  return outputDirPath

def faceMorph(detector, predictor, srcPath, targetPath, outputPath):
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
    outputImage = _faceMorph(srcImage, srcFaceFeatures, outputImage, targetFaceFeatures)
  writeImage(outputImage, outputPath)

def faceMorphInVideo(detector, predictor, srcPath, targetPath, outputPath):
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
      outputImage = _faceMorph(srcImage, srcFaceFeatures, outputImage, targetFaceFeatures)
    return outputImage
  
  videoprocessor = VideoProcessor(targetPath, outputPath, process)
  videoprocessor.start()

def _faceMorph(srcImage, srcFaceFeatures, targetImage, targetFacesWithFeatures):
  outputImage = face_swap_custom(srcImage, targetImage, srcFaceFeatures, targetFacesWithFeatures)
  return outputImage
