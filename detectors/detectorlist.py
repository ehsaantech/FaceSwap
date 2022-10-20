from detectors.dnnCaffe import DnnCaffeDetector
from detectors.hogSvm import HogSvmFaceDetector
from detectors.haarCascade import haarcascade_default, haarcascade_alt, haarcascade_alt2, haarcascade_alt_tree
from detectors.mediapipe import MediapipeDetector
from detectors.mmod import MModDetector
from detectors.mtcnn import MtcnnDetector

detectorList = [
  (HogSvmFaceDetector, 'hog_svm'),
  (MediapipeDetector, 'mediapipe'),
  (haarcascade_default, 'haar_default'),
  (haarcascade_alt, 'haar_alt'),
  (haarcascade_alt2, 'haar_alt1'),
  (haarcascade_alt_tree, 'haar_alt_tree'),
  (DnnCaffeDetector, 'dnn_caffe'),
  (MtcnnDetector, 'mtcnn'),
  (MModDetector, 'mmod'),
]