from utilities.image import readImage, writeImage
from utilities.videoprocessor import VideoProcessor
from images.imagelist import imageList as images
from videos.videolist import videoList as videos
from detectors.detectorlist import detectorList as detectors
from detectors.aggregatorslist import detectorList as aggregators

def compareDetectors(outputDirName) -> str:
  outputDirPath = '{0}/{1}'.format(outputDirName, 'detectors')

  _compareDetectors(detectors, outputDirPath)
  _compareDetectors(aggregators, outputDirPath)

  return outputDirPath

def _compareDetectors(detectors, outputDirPath):
  for (detectorClass, detectorName, _) in detectors:
    detector = detectorClass()
    for (imagePath, imageName, imageExtension) in images:
      outputPath = '{0}/{1}-{2}.{3}'.format(outputDirPath, imageName, detectorName, imageExtension)
      image = readImage(imagePath)
      faces = detector.detectFacesInImage(image)
      annotatedImage = detector.markFacesInImage(image, faces)
      writeImage(annotatedImage, outputPath)

    for (videoPath, videoName, videoExtension) in videos:
      outputPath = '{0}/{1}-{2}.{3}'.format(outputDirPath, videoName, detectorName, videoExtension)
      videoprocessor = VideoProcessor(videoPath, outputPath, detector.process)
      videoprocessor.start()