import logging

def takeCliOptionSelection(optionNames, optionType):
  print('Please choose a {} from the following:'.format(optionType))
  for index, name in enumerate(optionNames):
    print('{0} -> {1}'.format(index, name))
  
  selection = int(input('Type Option Number to select -> '))
  logging.debug('User selected {}'.format(selection))

  if selection >= 0 and selection < len(optionNames):
    return selection
  else:
    raise Exception('Invalid option -> {0} selected. Values should be in range 0-{1}'.format(selection, len(optionNames)-1))