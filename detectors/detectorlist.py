from detectors.dlibfrontalface import DlibFrontalFaceDetector
from detectors.mediapipe import MediapipeDetector
from detectors.mmod import MModDetector

detectorList = [
  (DlibFrontalFaceDetector, 'dlib'),
  (MediapipeDetector, 'mediapipe'),
  (MModDetector, 'mmod'),
]