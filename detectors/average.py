import functools
from pickle import FALSE
import dlib
import logging
from detectors.detector import Detector
from detectors.detectorlist import detectorList as detectors
from detectors.detectorProperties import minNumberOfDetections, detectionOverlapOffset

class AverageDetector(Detector):

  def __init__(self):
    logging.debug('initializing average detector')
    self.detectors = [className() for (className, _, _) in detectors]

  def reduce(self, accumulated, face):
    found = False
    for (index, (f, c)) in enumerate(accumulated):
      top = f.top()
      left = f.left()

      anchorLeft = left - detectionOverlapOffset
      anchorTop = top - detectionOverlapOffset
      anchorRight = left + detectionOverlapOffset
      anchorBottom = top + detectionOverlapOffset
      anchorRect = dlib.rectangle(left=anchorLeft, top=anchorTop, right=anchorRight, bottom=anchorBottom)

      if anchorRect.contains(face.tl_corner()):
        left = int(((f.left() * c) + face.left()) / (c + 1))
        top = int(((f.top() * c) + face.top()) / (c + 1))
        right = int(((f.right() * c) + face.right()) / (c + 1))
        bottom = int(((f.bottom() * c) + face.bottom()) / (c + 1))
        merged = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
        accumulated[index] = (merged, c + 1)
        found = True

    if not found:
      accumulated.append((face, 1))

    return accumulated


  def detectFacesInImage(self, img):
    faces = []
    for detector in self.detectors:
      detectedFaces = detector.detectFacesInImage(img)
      [faces.append(face) for face in detectedFaces]
    
    faces.sort(key=lambda k: [k.left(), k.top()], reverse=False)
    faces = functools.reduce(self.reduce, faces, [])
    faces = [face for (face, count) in faces if count >= minNumberOfDetections]
    logging.debug('Number of faces detected: {}'.format(len(faces)))
    return faces
