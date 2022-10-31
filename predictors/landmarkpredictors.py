import dlib
import logging
import numpy 
from predictors.predictor import Predictor

class LandmarkPredictor(Predictor):

  def __init__(self, predictor):
    logging.debug('initializing {}'.format(predictor))
    self.predictor = dlib.shape_predictor(predictor)

  def predictFeaturesInImage(self, img, faceRect):
    shape = self.predictor(img, faceRect)
    logging.debug("found {} shapes".format(shape.num_parts))
    coords = numpy.asarray([(p.x, p.y) for p in shape.parts()], dtype=int)
    return coords

Predictor_5 = lambda : LandmarkPredictor('predictors/models/shape_predictor_5_face_landmarks.dat') 
Predictor_68 = lambda : LandmarkPredictor('predictors/models/shape_predictor_68_face_landmarks.dat')
Predictor_68_GTX = lambda : LandmarkPredictor('predictors/models/shape_predictor_68_face_landmarks_GTX.dat')
Predictor_81 = lambda : LandmarkPredictor('predictors/models/shape_predictor_81_face_landmarks.dat')
    