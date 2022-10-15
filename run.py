#! /usr/bin/env python
import logging

from detectors.comparedetectors import compareDetectors
from predictors.comparepredictors import comparePredictors

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  selection = -1
  while (selection != 0):
    print('Welcome to Visionary!')
    print('Please choose an option from the following:')
    print('1 -> Compare Face Detectors')
    print('2 -> Compare Face Feature Predictors')
    print('3 -> Swap Face')
    print('4 -> Morph Face')
    print('0 -> Exit')
    selection = int(input('Type Option Number to select -> '))
    logging.debug('User selected {}'.format(selection))
    match selection:
      case 1:
        outputPath = compareDetectors(outputDirName='output')
        print('Please check {0} for output'.format(outputPath))
      case 2:
        outputPath = comparePredictors(outputDirName='output')
        print('Please check {0} for output'.format(outputPath))
      case 3:
        print('Face swapping')
      case 4:
        print('Face Morphing')
      case 0:
        print('Thank you for using Visionary')
      case _:
        print('Invalid option selected. Please try again.')
  
  
