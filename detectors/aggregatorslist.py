from detectors.all import AllDetectors
from detectors.average import AverageDetector
from detectors.merge import MergeDetector
from detectors.weightedaverage import WeightedAverageDetector

detectorList = [
  (MergeDetector, 'merge',1 ),
  (AllDetectors, 'all', 1),
  (AverageDetector, 'average', 1),
  (WeightedAverageDetector, 'weighted_average', 1)
]