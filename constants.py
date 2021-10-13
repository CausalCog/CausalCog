'''in_use_do_not_archive
constants.py module used with the CCA3

contains constants, many used in gdata.py, ndata.py, hdata.py

no classes at this time in this module


requirements.txt:
    provided simply here as a roadmap to the modules in the CCA3
    please check with cca4.py to make sure latest requirements

'''


##START PRAGMAS
#
#pylint: disable=line-too-long
#   prefer to take advantage of longer line length of modern monitors, even with multiple windows
#pylint: disable=invalid-name
#   prefer not to use snake_case style for very frequent data structure or small temp variables
#pylint: disable=bare-except
#   prefer in some code areas to catch any exception rising
#pylint: disable=too-many-branches
#pylint: disable=too-many-statements
#   prefer to use comfortable number of branches and statements, especially in user menu communication
#pylint: disable=too-many-instance-attributes
#pylint: disable=unused-wildcard-import
#pylint: disable=wildcard-import
#   use wildcard import for constants
##END PRAGMAS


## START IMPORTS   START IMPORTS
#
##standard imports -- being used by this module
try:
    #import pdb
    import sys
    #import platform
    #import os.path
    #import random
    #import copy
except ImportError:
    print('\nprogram will end -- constants.py module of causal cog arch unable to import standard lib module')
    print('please ensure correct version of python can be accessed')
    sys.exit()
#
##PyPI imports -- being used by this module
try:
    #import numpy as np
    #import colorama # type: ignore
    #import pyfiglet # type: ignore
    #import termcolor
    pass
except ImportError:
    print('\nprogram will end -- constants.py module of the causal cog arch unable to import a PyPI module')
    print('please check requirements.txt and install all required dependencies')
    sys.exit()
#
##non-PyPI third-party imports -- being used by this module
try:
    pass
    #justification/ Awesome/LibHunt ratings for non-pypi imports: n/a
    #nb. none
except ImportError:
    print('program will end -- constants.py module of the causal cog arch unable to import a third-party module')
    print('please check requirements.txt and install all required dependencies')
    sys.exit()
#
##CCA1 module imports -- being used by this module
try:
    #from constants import *
    #import gdata
    #import ddata
    ##import hdata
    #import main_mech
    #import eval_micro  #June 2021 deprecated
    #import eval_milli  #June 2021 deprecated
    #import palimpsest  #nb  without GPU will use excessive resources
    pass
except ImportError:
    print('program will end -- constants.py module unable to import a causal cognitive architecture module')
    print('please check requirements.txt and install all required dependencies')
    sys.exit()
#
#


##START CONSTANTS
#
VERSION = 'not specified'
HARDWARE = False
MEMORY_CHECKING_ON_TEMP = False
FULL_CAUSAL = False
BINDING = True #version for CCA3 Binding Paper to avoid GPU, demonstrate equations
DEBUG = True
FASTRUN = True #True causes skipping of many user inputs
AUTORUN = False #True will run whole session without user input
LIFESPAN = 10000  #max loops for main_eval()
MOD_CYCLE_REEVALUATE = 5
TOTAL_ROWS = 6  #count EDGE squares
TOTAL_COLS = 6  #count EDGE squares
GOAL_RANDOM_WALK = '00000000'
GOAL_SKEWED_WALK = '00000001'
GOAL_PRECAUSAL_FIND_HIKER = '11111111'
GOAL_CAUSAL_FIND_HIKER = '11110000'
TRIES_BEFORE_DECLARE_LOCAL_MINIMUM = 2
DEFAULT_VECTOR = '00000000'
DEFAULT_GOAL = GOAL_RANDOM_WALK
DEFAULT_HIPPOCAMPUS = 'HUMAN'
DEFAULT_FIRST_SCENE = 'FOREST'
ESCAPE_LEFT = '11111111'
FILLER = '00000000'
REFLEX_ESCAPE = '10011001'
INITIATE_VALUE = 0
FIRST_SCENE = 'MOTHER'
CONTINUATION_TEXT = 'Please press ENTER to continue....'
MISSION_COUNTER = 0
MAX_CYCLES_NOW_EXIT = 20
TOTAL_MAPS =1000
TOTAL_OBJECTS = 11  #segments 0-10
TOTAL_ENVIRONMENTS = 1000
TOTAL_SCENES = 20
TOTAL_STREAMS = 20
STANDARD_DELAY = 2
##END CONSTANTS
