import logging
from utilities.cli import takeCliOptionSelection
from utilities.image import readImage, writeImage
from utilities.videoprocessor import VideoProcessor
from images.imagelist import imageList as images
from videos.videolist import videoList as videos
from detectors.detectorlist import detectorList as detectors
from predictors.predictorlist import predictorList as predictors

def comparePredictorsCli(outputDirName) -> str:
  outputDirPath = '{0}/{1}'.format(outputDirName, 'predictors')
  try:
    selection = takeCliOptionSelection([d[1] for d in detectors], 'detector')
    comparePredictors(detectors[selection], outputDirPath)
  except Exception as exception:
    logging.warning(exception)
    return

  return outputDirPath

def comparePredictors(detector, outputDirPath):
  (detectorClass, detectorName) = detector
  detector = detectorClass()
  for (predictorClass, predictorName) in predictors:
    predictor = predictorClass()
    for (imagePath, imageName, imageExtension) in images:
      outputPath = '{0}/{1}-{2}-{3}.{4}'.format(outputDirPath, imageName, predictorName, detectorName, imageExtension)
      image = readImage(imagePath)
      faces = detector.detectFacesInImage(image)
      facesWithFeatures = [predictor.predictFeaturesInImage(image, face) for face in faces]
      annotatedImage = predictor.markFeaturesInImage(image, facesWithFeatures)
      writeImage(annotatedImage, outputPath)

      outputPath = '{0}/{1}-hull-{2}-{3}.{4}'.format(outputDirPath, imageName, predictorName, detectorName, imageExtension)
      annotatedImage = predictor.markFaceHullsInImage(image, facesWithFeatures)
      writeImage(annotatedImage, outputPath)

    def process(image):
      faces = detector.detectFacesInImage(image)
      facesWithFeatures = [predictor.predictFeaturesInImage(image, face) for face in faces]
      return predictor.markFeaturesInImage(image, facesWithFeatures)

    def processHull(image):
      faces = detector.detectFacesInImage(image)
      facesWithFeatures = [predictor.predictFeaturesInImage(image, face) for face in faces]
      return predictor.markFaceHullsInImage(image, facesWithFeatures)

    for (videoPath, videoName, videoExtension) in videos:
      outputPath = '{0}/{1}-{2}-{3}.{4}'.format(outputDirPath, videoName, predictorName, detectorName, videoExtension)
      videoprocessor = VideoProcessor(videoPath, outputPath, process)
      videoprocessor.start()

      outputPath = '{0}/{1}-hull-{2}-{3}.{4}'.format(outputDirPath, videoName, predictorName, detectorName, videoExtension)
      videoprocessor = VideoProcessor(videoPath, outputPath, processHull)
      videoprocessor.start()
