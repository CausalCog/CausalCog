'''in_use_do_not_archive
gdata.py module used with the CCA3

contains class MultipleSessionsData -- instantiation of this class
gdata.py is kept between CCA3 missions

    g = gdata.MultipleSessionsData() #g persists between missions, initialized in cca3.main_eval()
    d = ddata.MapData() #d re-initialized between missions always
    h = hdata.NavMod()  #h re-initialized between missions optionally via choose_simulation()
    m = hdata.MapFeatures() #m re-initialized between missions optionally via choose_simulation()

variety of helper data structures, data and related methods are included
in this class

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
#pylint: disable=too-many-instance-attributes
#   prefer to use comfortable number of branches and statements, especially in user menu communication
#pylint: disable=unused-wildcard-import
#pylint: disable=wildcard-import
#   use wildcard import for constants
##END PRAGMAS


# START IMPORTS   START IMPORTS
#
##standard imports -- being used by this module
try:
    #import pdb
    import sys
    import time
    #import platform
    import os.path
    #import random
    #import copy
    from PIL import Image  # type: ignore
except ImportError:
    print('\nprogram will end -- gdata module of causal cog arch unable to import standard lib module')
    print('please ensure correct version of python can be accessed')
    sys.exit()
#
##PyPI imports -- being used by this module
try:
    #import numpy as np
    import colorama     # type: ignore
    import pyfiglet     # type: ignore
    import termcolor    # type: ignore
    from termcolor import colored
except ImportError:
    print('\nprogram will end -- gdata module of the causal cog arch unable to import a PyPI module')
    print('please check requirements.txt and install all required dependencies')
    sys.exit()
#
##non-PyPI third-party imports -- being used by this module
try:
    pass
    #justification/ Awesome/LibHunt ratings for non-pypi imports: n/a
    #nb. none
except ImportError:
    print('program will end -- gdata module of the causal cog arch unable to import a third-party module')
    print('please check requirements.txt and install all required dependencies')
    sys.exit()
#
##CCA1 module imports -- being used by this module
try:
    from constants import *
    #import gdata
    #import ddata
    ##import hdata
    #import main_mech
    #import eval_micro  #June 2021 deprecated
    #import eval_milli  #June 2021 deprecated
    #import palimpsest  #nb  without GPU will use excessive resources
except ImportError:
    print('program will end -- gdata.py module unable to import a causal cognitive architecture module')
    print('please check requirements.txt and install all required dependencies')
    sys.exit()
#
#


class MultipleSessionsData:
    '''in_use_do_not_archive
    hold data here that should be kept between sessions as well
    as global constants
    '''
    def __init__(self):
        '''in_use_do_not_archive
            the following are initiated at program start via instant-
            iation of ddata.MultipleSessionsData -- 'g' instance:
        '''
        #print('debug: MultipleSessionsData is being instantiated -- if to g make sure appropriate -- data will be erased')
        self.raw_future_expected_values_from_multiple_missions: list = [['$d.current_goal', '$h.exit_reason', '$d.raw_future_expected_value_for_this_mission']]
        self.event_log_memory = [['start of event_log_memory']]
        self.sensory_buffer: list = []
        self.debug = DEBUG
        self.fastrun = FASTRUN #True causes skipping of many user inputs
        self.autorun = AUTORUN #True will run whole session without user input
        self.mod_cycle_reevaluate = MOD_CYCLE_REEVALUATE
        self.version = VERSION
        self.hardware = HARDWARE
        self.memory_checking_on_temp = MEMORY_CHECKING_ON_TEMP
        self.total_rows = TOTAL_ROWS
        self.total_cols = TOTAL_COLS
        self.goal_default = []
        self.goal_random_walk = GOAL_RANDOM_WALK
        self.goal_skewed_walk = GOAL_SKEWED_WALK
        self.goal_precausal_find_hiker = GOAL_PRECAUSAL_FIND_HIKER
        self.goal_causal_find_hiker = GOAL_CAUSAL_FIND_HIKER
        self.tries_before_declare_local_minimum = TRIES_BEFORE_DECLARE_LOCAL_MINIMUM
        #self.default_vector = DEFAULT_VECTOR
        #self.default_goal = DEFAULT_GOAL
        #self.default_hippocampus = DEFAULT_HIPPOCAMPUS
        self.escape_left = ESCAPE_LEFT
        self.filler = FILLER
        self.reflex_escape = REFLEX_ESCAPE
        #self.initiate_value = INITIATE_VALUE
        self.continuation_text = CONTINUATION_TEXT
        self.mission_counter = MISSION_COUNTER
        self.max_evln_cycles_now_exit = MAX_CYCLES_NOW_EXIT
        self.intermediate_results = False
        self.standard_delay = STANDARD_DELAY


    def pass_g(self)-> bool:
        '''in_use_do_not_archive
        this function executes a pass-equivalent statement and does nothing else
        purpose is to include for scaffolding reasons this class can be
        instantiated in the calling module and not generate a pylint error
        '''
        if self.debug == 'error':
            print('instance.debug == error')
        return True


    def __str__(self)-> str:
        '''
        in_use_do_not_archive
        for developmental purposes
        values of the instance.MultipleSessionsData values
        '''
        print('\n*******dump: instance.MultipleSessionsData variables*****')
        print('self.raw_future_expected_values_from_multiple_missions  ', self.raw_future_expected_values_from_multiple_missions)
        print('self.event_log_memory  ', self.event_log_memory)
        print('self.sensory_buffer  ', self.sensory_buffer)
        print('self.debug  ', self.debug)
        print('self.fastrun  ', self.fastrun)
        print('self.autorun', self.autorun)
        print('self.mod_cycle_reevaluate  ', self.mod_cycle_reevaluate)
        print('self.version  ', self.version)
        print('self.max_evln_cycles_now_exit', self.max_evln_cycles_now_exit)
        print('self.memory_checking_on_temp  ', self.memory_checking_on_temp)
        print('for other variables change __str__ or parameters if possible')
        return '*******finished dump: instance.MultipleSessionsData variables*****\n'


    def printout_event_log_memory(self)->bool:
        '''in_use_do_not_archive
        prints event_log memory and whatever analysis method provides
        '''
        print('\n', self.event_log_memory, '\n')
        return True


    def gevent_log(self, item: str, verbose: bool = False)->bool:
        '''to deprecate or rewrite for cca4.py
        goal and event_log module interacts with the emotional and reward module as well
         as the entire CCA1 to provide some overall control of the CCA1’s behavior
         memories of operations occurring in the logic/working memory are temporarily
          kept in the event_log module, allowing improved problem solving as well as
          providing more transparency to CCA1 decision making
        '''
        if verbose:
            print('CHECKPOINT: in event_log method')
        #nano ver emulate with simple list
        #add event_log item to event_log memory
        self.event_log_memory.append(item)
        print('in gevent_log')
        return True


    def show_architecture_related(self, imagename: str, part: str = "architecture")-> bool:
        '''in_use_do_not_archive
        shows diagram of Causal Cognitive Architecture components to better let
        user follow what stages the software is going through'''
        if self.fastrun:
            return True
        print(f"\nWould you like to see a diagram of the {part} to better follow what modules the code is working through??")
        print("\n(Note: Any images displayed below will be outside of terminal. If you decide to view any images, after you exit")
        print("the image, you will return back to the CCA3 program.\n")
        try:
            b_b = input("Enter 'Y' or 'y' to see this diagram (just hit 'ENTER' to ignore): ")
        except:
            b_b = "not valid"
        if b_b in ("y", "Y", "yes", "Yes"):
            try:
                image = Image.open(imagename)
                image.show()
                print('\n')
                return True
            except:
                print("unable to display this image (perhaps image is not present or platform issues)\n")
                return False
        return True


    def large_letts_display(self, to_print: str, numeric_value: int=1)->bool:
        '''in_use_do_not_archive
        display ascii-text art
        currently prints in white
        '''
        if self.fastrun:
            #return False
            pass
        print()
        try:
            colorama.init(strip=not sys.stdout.isatty()) #do not use colors if stdout
            termcolor.cprint(pyfiglet.figlet_format(to_print), color='white', attrs=['bold'])
            if numeric_value%5 == 0:
                self.ascii_art_display('spaceship')
            return True
        except:
            print('\n', to_print)
            print('large letters/color message did not display\n')
            return False


    def ascii_art_display(self, to_print: str)->bool:
        '''in_use_do_not_archive
        display ascii-text art
        currently prints in same color as monitor text
        '''
        #setup
        if self.fastrun:
            #return False
            pass

        #spaceship
        if to_print == 'spaceship':
            print('''


                        /\\
                       /  \\
                      |    |
                      |fast|
                      |code|
                      |    |
                      |    |
                     '      `
                     |      |
                     |      |
                     |______|
                      '-`'-`   .
                      / . \\'\\ . .'
                     ''( .'\\.' ' .;'
                    '.;.;' ;'.;' ..;;'

            ''')
            return True

        #brain
        if to_print == 'brain':
            print('this is a brain -- image not in code yet')
            return True

        print('debug: unable to print ascii_art_display: ', to_print)
        return False


    def one_moment_please_display(self, delay: int = -1)-> bool:
        '''in_use_do_not_archive
        as program gets larger and more libraries and codebases will need to be
         initialized at the start of the program this message asking user to wait
         may be useful
        currently small delay function, however, consider replacing time.sleep() with
         the initialization routines and once complete display ready message
        displays ascii-text art message that there is a delay
        currently prints large message in white, art and caption in same color as monitor text
        "One  moment  please...." message
        input parameters:
           delay -- display image for "delay" number of seconds
                    if negative number or default -1 then becomes standard_delay of
                    2 seconds (or check above what current value is)
        returns:
           True/False
        '''
        #pylint: disable=anomalous-backslash-in-string
        os.system("cls")
        if BINDING:
            self.large_letts_display("CCA3  DEMONSTRATION")
        else:
            self.large_letts_display("CCA3")
        print("""                                  ,
                              ,   |
           _,,._              |  0'
         ,'     `.__,--.     0'
        /   .--.        |           ,,,
        | [=========|==|==|=|==|=|==___]
        \   "--"  __    |           '''
         `._   _,'  `--'
            ""'
          ____
          |  |
         0' 0'

        One moment while we tune up the code for you....
        """)
        self.startup_overhead(2*delay)
        self.large_letts_display("Ready")
        input('        Ready.... please press ENTER to continue....')
        os.system("cls")
        #pylint: enable=anomalous-backslash-in-string
        return True


    def startup_overhead(self, delay, option=1):
        '''in_use_do_not_archive
        start of program to set up large arrays, initialized
        neural networks, etc
        if BINDING == True set for CCA4 demonstration version that does not required GPU
        '''
        if delay <0:
            delay = self.standard_delay
        time.sleep(delay)
        if (option == 4 or BINDING is True):
            self.large_letts_display("INSTRUCTIONAL DEMO  TO  USE WITH  CCA3 BINDING  PAPER")
            print(colored('Blue letters like this are specifically for the CCA3 Binding paper', 'cyan'))
            print(colored("NOTE:  ALL EQUATION NUMBERS LINKED AND DEMONSTRATED", "cyan"))
            print(colored("NOTE:  NO GPU REQUIRED FOR THIS VERSION -- FUZZYWUZZY USED FOR PATTERN RECOGNITION", 'cyan'))
            print(colored("(navigation maps will be updated, but some of the recognition learning will not occur)", 'cyan'))
            return True
        if option == 1:
            # GPU appropriate library required
            print("Local or cloud GPU checking not enabled at present")
            print("CPU may possibly run routines or else failure will occur")
            print("program execution continues but failure may occur....")
            return False
        print("\ndebug: startup_overhead incorrect argument")
        return False


    def choose_if_g_fastrun_on_off(self):
        '''in use do not archive
        user can choose at start of program the verbosity of messages
        useful for development work
        '''
        self.ascii_art_display("spaceship")
        self.large_letts_display("Press   ENTER\n( or   rocket   code )\n")
        print('To run simulation in normal way -- press ENTER\n')
        print('To run simulation in quicker developer mode (removes many prompts) -- "d" + ENTER\n')
        print('Other developer "rocket codes" pending\n')
        try:
            dev_code = input("Press ENTER to continue (or developer code): ")
            if dev_code in ['', 'off', 'OFF', 'normal']:
                return_value = self.toggle_g_fastrun_off()
            else:
                print(f'Development code {dev_code} entered')
                return_value = self.toggle_g_fastrun_on()
                if dev_code in ['-1', 'special']:
                    input('Development mode as note above selected.... press ENTER to continue....')
        except:
            print('debug: invalid input -- g.fastrun will be False')
            return_value = self.toggle_g_fastrun_off()
        return return_value


    def toggle_g_fastrun_off(self):
        '''in_use_do_not_archive
        turn g.fastrun to False
        '''
        self.fastrun = False
        #print('g.fastrun is: ', self.fastrun) #keep screen clean and avoid print if off
        return False


    def toggle_g_fastrun_on(self):
        '''in_use_do_not_archive
        turn g.fastrun to False
        '''
        self.fastrun = True
        print('g.fastrun is: ', self.fastrun)
        return True


    def check_if_g_fastrun_on(self):
        '''in_use_do_not_archive
        checks to see if g_fastrun and gives warning that system is
        in a development mode, ie, will be running past sections of input code
        '''
        if not self.fastrun:
            return False #keep screen clean and avoid print if off
        input('g.fastrun is set to True for development mode.... press ENTER to continue')
        return True


    def fast_input(self, to_print: str=' ', temp_off=False)->bool:
        '''in_use_do_not_archive
        use instead of input() to allow user a chance to look at screen and then
        press ENTER to continue
           advantage is that if g.fastrun is true, then input() is bypassed
           useful to allow user to go through program more quickly or for automated runs of program
        '''
        if (self.fastrun and not temp_off):
            return False
        if to_print == '':
            to_print = self.continuation_text  #'Please press ENTER to continue....'
        input(to_print)
        return True
