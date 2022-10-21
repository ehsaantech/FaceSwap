import functools
import dlib
import logging
from detectors.detector import Detector
from detectors.detectorlist import detectorList as detectors
from detectors.detectorProperties import minNumberOfDetections, detectionOverlapOffset

class MergeDetector(Detector):

  def __init__(self):
    logging.debug('initializing merge detector')
    self.detectors = [className() for (className, _, _) in detectors]

  # def detectFacesInImage(self, img):
  #   faces = []
  #   for detector in self.detectors:
  #     detectedFaces = detector.detectFacesInImage(img)

  #     if len(faces) == 0:
  #       faces = [ (face, 1) for face in detectedFaces]
  #       continue
      
  #     for face in detectedFaces:
  #       top = face.top()
  #       left = face.left()

  #       anchorLeft = left - detectionOverlapOffset
  #       anchorTop = top - detectionOverlapOffset
  #       anchorRight = left + detectionOverlapOffset
  #       anchorBottom = top + detectionOverlapOffset
        
  #       anchorRect = dlib.rectangle(left=anchorLeft, top=anchorTop, right=anchorRight, bottom=anchorBottom)
  #       found = False
  #       for (index, (f, count)) in enumerate(faces):
  #         if anchorRect.contains(f.tl_corner()):
  #           left = min(face.left(), f.left())
  #           top = min(face.top(), f.top())
  #           right = max(face.right(), f.right())
  #           bottom = max(face.bottom(), f.bottom())
  #           merged = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
  #           faces[index] = (merged, count + 1)
  #           found = True

  #       if not found:
  #         faces.append((face, 1))

  #   faces = [face for (face, count) in faces if count >= minNumberOfDetections]
  #   logging.debug('Number of faces detected: {}'.format(len(faces)))
  #   return faces


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
        left = min(face.left(), f.left())
        top = min(face.top(), f.top())
        right = max(face.right(), f.right())
        bottom = max(face.bottom(), f.bottom())
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
