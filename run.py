#! /usr/bin/env python
import logging

from detectors.comparedetectors import compareDetectors
from faceswap import faceswapCli
from facemorph import morphCli
from predictors.comparepredictors import comparePredictorsCli

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
    if selection == 1:
      outputPath = compareDetectors(outputDirName='output')
      print('Please check {} for output'.format(outputPath))
    elif selection == 2:
      outputPath = comparePredictorsCli(outputDirName='output')
      print('Please check {} for output'.format(outputPath))
    elif selection == 3:
      outputPath = faceswapCli(outputDirName='output')
      print('Please check {} for output'.format(outputPath))
    elif selection == 4:
      outputPath = morphCli(outputDirName='output')
      print('Please check {} for output'.format(outputPath))
    elif selection == 0:
      print('Thank you for using Visionary')
    else:
      print('Invalid option selected. Please try again.')
  
  
