import functools
from pickle import FALSE
import dlib
import logging
from detectors.detector import Detector
from detectors.detectorlist import detectorList as detectors
from detectors.detectorProperties import minWeightOfDetections, detectionOverlapOffset

class WeightedAverageDetector(Detector):

  def __init__(self):
    logging.debug('initializing average detector')
    self.detectors = [(className(), weight) for (className, _, weight) in detectors]

  def reduce(self, accumulated, current):
    found = False
    (face, weight) = current
    for (index, (f, w)) in enumerate(accumulated):
      top = f.top()
      left = f.left()

      anchorLeft = left - detectionOverlapOffset
      anchorTop = top - detectionOverlapOffset
      anchorRight = left + detectionOverlapOffset
      anchorBottom = top + detectionOverlapOffset
      anchorRect = dlib.rectangle(left=anchorLeft, top=anchorTop, right=anchorRight, bottom=anchorBottom)

      if anchorRect.contains(face.tl_corner()):
        totalWeight = w + weight
        left = int(((f.left() * w) + (face.left() * weight)) / totalWeight)
        top = int(((f.top() * w) + (face.top() * weight)) / totalWeight)
        right = int(((f.right() * w) + (face.right() * weight)) / totalWeight)
        bottom = int(((f.bottom() * w) + (face.bottom() * weight)) / totalWeight)
        merged = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
        accumulated[index] = (merged, totalWeight)
        found = True

    if not found:
      accumulated.append((face, weight))

    return accumulated


  def detectFacesInImage(self, img):
    faces = []
    for (detector, weight) in self.detectors:
      detectedFaces = detector.detectFacesInImage(img)
      [faces.append((face, weight)) for face in detectedFaces]
    
    faces.sort(key=lambda k: [k[0].left(), k[0].top()], reverse=False)
    faces = functools.reduce(self.reduce, faces, [])
    faces = [face for (face, weight) in faces if weight >= minWeightOfDetections]
    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return faces
