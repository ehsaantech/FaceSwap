import logging
from utilities.cli import takeCliOptionSelection
from utilities.image import readImage, writeImage
from utilities.videoprocessor import VideoProcessor
from images.imagelist import imageList as images
from videos.videolist import videoList as videos
from detectors.detectorlist import detectorList as detectors
from detectors.aggregatorslist import detectorList as aggregators
from predictors.predictorlist import predictorList as predictors

def comparePredictorsCli(outputDirName) -> str:
  outputDirPath = '{0}/{1}'.format(outputDirName, 'predictors')
  try:
    options = detectors + aggregators
    detectorSelection = takeCliOptionSelection([d[1] for d in options], 'detector')
    predictorSelection = takeCliOptionSelection([p[1] for p in predictors], 'predictor')
    _comparePredictors(options[detectorSelection], predictors[predictorSelection], outputDirPath)
  except Exception as exception:
    logging.warning(exception)
    return

  return outputDirPath

def _comparePredictors(detector, predictor, outputDirPath):
  (detectorClass, detectorName, _) = detector
  detector = detectorClass()
  (predictorClass, predictorName) = predictor
  predictor = predictorClass()  

  for (imagePath, imageName, imageExtension) in images:
    outputPath = '{0}/{1}-{2}-{3}.{4}'.format(outputDirPath, imageName, predictorName, detectorName, imageExtension)
    image = readImage(imagePath)
    faces = detector.detectFacesInImage(image)
    facesWithFeatures = [predictor.predictFeaturesInImage(image, face) for face in faces]
    annotatedImage = predictor.markFeaturesInImage(image, facesWithFeatures)
    writeImage(annotatedImage, outputPath)

    # outputPath = '{0}/{1}-hull-{2}-{3}.{4}'.format(outputDirPath, imageName, predictorName, detectorName, imageExtension)
    # annotatedImage = predictor.markFaceHullsInImage(image, facesWithFeatures)
    # writeImage(annotatedImage, outputPath)

  def process(image):
    faces = detector.detectFacesInImage(image)
    facesWithFeatures = [predictor.predictFeaturesInImage(image, face) for face in faces]
    return predictor.markFeaturesInImage(image, facesWithFeatures)

  # def processHull(image):
  #   faces = detector.detectFacesInImage(image)
  #   facesWithFeatures = [predictor.predictFeaturesInImage(image, face) for face in faces]
  #   return predictor.markFaceHullsInImage(image, facesWithFeatures)

  for (videoPath, videoName, videoExtension) in videos:
    outputPath = '{0}/{1}-{2}-{3}.{4}'.format(outputDirPath, videoName, predictorName, detectorName, videoExtension)
    videoprocessor = VideoProcessor(videoPath, outputPath, process)
    videoprocessor.start()

    # outputPath = '{0}/{1}-hull-{2}-{3}.{4}'.format(outputDirPath, videoName, predictorName, detectorName, videoExtension)
    # videoprocessor = VideoProcessor(videoPath, outputPath, processHull)
    # videoprocessor.start()
