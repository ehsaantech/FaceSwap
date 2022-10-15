import logging
from utilities.image import readImage, writeImage
from utilities.videoprocessor import VideoProcessor
from images.imagelist import imageList
from videos.videolist import videoList
from detectors.detectorlist import detectorList
from predictors.predictorlist import predictorList

images = imageList
videos = videoList
detectors = detectorList
predictors = predictorList

def comparePredictors(outputDirName) -> str:
  print('Please choose a detector from the following:')
  for index, (_, detectorName) in enumerate(detectors):
    print('{0} -> {1}'.format(index, detectorName))
  
  selection = int(input('Type Option Number to select -> '))
  logging.debug('User selected {}'.format(selection))

  if selection >= 0 and selection < len(detectors):
    _comparePredictors(outputDirName, detectors[selection])
  else:
    return 'Incorrect selection value for detector. Aborting'

def _comparePredictors(outputDirName, detector) -> str:
  (detectorClass, detectorName) = detector
  outputDirPath = '{0}/{1}'.format(outputDirName, 'predictors')
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

  return outputDirPath
