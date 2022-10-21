from detectors.dnnCaffe import DnnCaffeDetector
from detectors.hogSvm import HogSvmFaceDetector
from detectors.haarCascade import haarcascade_default, haarcascade_alt, haarcascade_alt2, haarcascade_alt_tree
from detectors.mediapipe import MediapipeDetector
from detectors.mmod import MModDetector
from detectors.mtcnn import MtcnnDetector

detectorList = [
  (HogSvmFaceDetector, 'hog_svm', 1),
  (MediapipeDetector, 'mediapipe', 1),
  (haarcascade_default, 'haar_default', 1),
  (haarcascade_alt, 'haar_alt', 1),
  (haarcascade_alt2, 'haar_alt1', 1),
  (haarcascade_alt_tree, 'haar_alt_tree', 1),
  (DnnCaffeDetector, 'dnn_caffe', 2),
  (MtcnnDetector, 'mtcnn', 3),
  (MModDetector, 'mmod', 1),
]