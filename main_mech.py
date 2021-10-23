#!/usr/bin/env python
#pylint: disable=line-too-long
'''in_use_do_not_archive
main_mech.py


Causal Cognitive Architecture 3 (CCA3)
June 2021 full rewrite (all previous code deprecated)
Oct 2021 CCA3 Binding Solution paper demonstration version

Note: Where you see "cca3.py" the module for the demonstration version is actually
"cca4.py" (cca3.py and cca4.py do the same thing but cca4.py is the rewrite for this version)

"evaluation cycle" == "processing cycle" == "cycle" == "loop through the architecture"
("machine cycle" is not used as a number of machine cycles depending on hardward for each evaluation cycle)

please see papers listed in cca3.py module docstring for more details
about the operation of the CCA3

cycles(d, g, h, m):
input parameters:
    d, g, h, m (data and method structures instantiations)
returns:
    #returns d, g, h, m since they are modified by this method

functional overview of python modules:
    cca3.py - intro screens, user enters hyperparameters, links to relevant versions of simulations
    main_mech.py - runs CCA3 evaluation cycles
    hdata.py - much of latest simulation is in the NavMod class, instantiated as 'h' in cca3.py
    ddata.py - instantiated as 'd' in cca3.py, used previous simulations but portions still used
    gdata.py - instantiated as 'g' in cca3.py, used previous simulations but portions still used
    constants.py - holds constants used by other modules

requirements.txt:
    provided simply here as a roadmap to the modules in the CCA3
    please check with cca4.py to make sure latest requirements
'''

##START PRAGMAS
#
# pylint: disable=invalid-name
#   prefer not to use snake_case style for very frequent data structure or small temp variables
# pylint: disable=bare-except
#   prefer in some code areas to catch any exception rising
# pylint: disable=too-many-lines
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-locals
#   prefer to use comfortable number of branches and statements, especially in user menu communication
# pylint: disable=pointless-string-statement
#
##END PRAGMAS

## START IMPORTS   START IMPORTS
#
##standard imports -- being used by this module
import random
import sys
# import os.path
# import copy
import msvcrt
# import math
# import pdb
# import time
# import pickle
# import threading
# import logging
# from io import StringIO
#
##PyPI imports -- being used by this module
try:
    import numpy as np  # type: ignore
    #  justificaton: AwesomePython 9.7, industry standard library
    # import pytorch
    #  justificaton: AwesomePython 9.9, industry standard library
    # from pympler import muppy
    #  justification: GitHub 525 stars, last commit 6 mos ago
    # from fuzzywuzzy import fuzz  #type: ignore
    from fuzzywuzzy import process  # type: ignore
    #  justification: AwesomePython 8.9
    from termcolor import colored
    #  justification: AwesomePython not rated; libraries.io -- 0 dependencies, stable since 2011
    #from icecream import ic  # type: ignore
    #justification: Awesome rating 7.9

except ImportError:
    print("\nprogram will end -- main_mech.py module unable to import a PyPI module")
    print("please check requirements.txt and install all required dependencies")
    sys.exit()
#
##non-PyPI third-party imports -- being used by this module
try:
    pass
    # justification/ Awesome/LibHunt ratings for non-pypi imports: n/a
except ImportError:
    print(
        "program will end -- main_mech.py module unable to import a third-party module"
    )
    print("please check requirements.txt and install all required dependencies")
    sys.exit()
#
##CCA3 module imports -- being used by this module
try:
    from constants import BINDING
    # import hdata
    # from constants import *
    # import palimpsest  #deprecated June 2021
except ImportError:
    print("program will end -- main_mech.py module unable to import a cca3 module")
    print("please check requirements.txt and install all required dependencies")
    sys.exit()
#
###END IMPORTS          END IMPORTS


##START MAIN_MECH.CYCLES()
#
def cycles(d, g, h, m):
    '''in_use_do_not_archive
    loop of evaluation cycles until the CCA3 completes or fails or
    times out of its mission/scene, and then returns back to the calling method cca3.main_eval()

    -'evaluation cycles' == 'processing cycles' == 'cycles' == loop through the architecture
    sensory --> processing --> output
    -'sensory scene' -- the visual, auditory, etc sensory inputs presented to the cca3 agent
    -note: sensory input data can stay the same the next evaluation cycle (i.e., can have multiple evaluation
    cycles for one scene of sensory input data) or perhaps there will be new sensory input scene the next evaluation cycle

    -cycles(d, g, h, m) is called by cca3.main_eval(): loop:
        input parameters:
            d, g, h, m (data and method structures instantiations)
        returns:
            #returns d, g, h, m since they are modified by this method
    '''
    # START of EVALN CYCLE LOOP
    # -->SENSORY INPUTS -> CCA3 -> MOTOR OUTPUTS ---REPEAT-^
    h.exit_cycles = False
    for d.evaluation_cycles in range(sys.maxsize):
        display_cycle_info(d, g, h)
        autonomic_check(g)
        next_scene_from_envrt = (h.envrt_interaction_and_input_sensory_vectors_shaping_modules(g))
        h.input_sensory_vectors_association_modules(g)
        h.sequential_error_correcting_module(g)
        h.object_segmentation_module(g)
        h.navigation_module(d, g)
        h.output_vector_assocation_module(d, g, '', '', 0, 0, 0, -1)
        if decide_to_exit(next_scene_from_envrt, d, g, h):
            d, g, h = update_expected_values(d, g, h)
            return d, g, h, m
        # LOOP AGAIN  loop for another evaluation cycle --------------^
    return None
##END MAIN_MECH.CYCLES()


##START OTHER METHODS     START OTHER METHODS
#
def display_cycle_info(d, g, h):
    '''in_use_do_not_archive
    displays evaluation cycle statistics to the user
    '''
    if BINDING:
        print(colored('\nCycles of Equations will now start', 'cyan'))
        print(colored('Recall that a "cycle" is a cycle of all the equations', 'cyan'))
        print(colored('Then in the next cycle, the equations repeat although not re-initialized', 'cyan'))
        print(colored('"Scenes" (i.e., "sensory scenes") are a new set of visual, auditory, etc', 'cyan'))
        print(colored('stimuli being presented to the CCA3. Sensing of a new scene occurs at the', 'cyan'))
        print(colored('start of the equations, i.e., with a new cycle. However.... cycles can repeat', 'cyan'))
        print(colored('while the same sensory scene is there, ie, can have multiple cycles each sensory scene', 'cyan'))
    if h.current_sensory_scene < 0:
        print("resetting a negative completed environment back to 0th scene")
        h.current_sensory_scene = 0
    print(f"\nSTARTING EVALUATION CYCLE # {d.evaluation_cycles} (run # {g.mission_counter} since simulation started)")
    g.large_letts_display("Cycle #  " + str(d.evaluation_cycles))
    g.large_letts_display("Scene #  " + str(h.current_sensory_scene))
    g.show_architecture_related("cca3_architecture.jpg", "CCA3 architecture")
    #g.fast_input("Please press ENTER to continue", True)
    return True


def decide_to_exit(next_scene_from_envrt, d, g, h, one_loop = True):
    '''in_use_do_not_archive
    makes decision to exit from main_mech.cycles() main loop
    '''
    #one_loop mode is intended to allow one loop through cycle() and then return back to main_eval()
    if one_loop:
        return True
    #exit if no more scenes to be served, i.e., 'mission' in this group of sensory scenes is complete
    if next_scene_from_envrt < 0:
        h.exit_reason = " no more scenes to be served"
        return True
    #exit if h.exit_cycles becomes set (used previously with cca1 after completion of a mission, but still may be in use in some routines)
    if h.exit_cycles:
        h.exit_reason = " h.exit_cycles was set"
        return True
    #exit if have already reached the maximum cycles specified by g.max_evln_cycles_now_exit (e.g., 20 cycles or whatever setting is at this time)
    if d.evaluation_cycles > g.max_evln_cycles_now_exit:
        h.exit_reason = "d.evaluation_cycles > g.max_evln_cycles_now_exit"
        return True
    #exit if have almost reached sys.maxsize - 100 which actually is 9,223,372,036,854,775,707 in current test
    if d.evaluation_cycles > sys.maxsize - 100:
        h.exit_reason = "d.evaluation_cycles > sys.maxsize - 100"
        return True
    #otherwise return False so exit and return from main_mech.cycles() will not occur now
    return False


def setup_user_view(d, g, h) -> bool:
    '''CCA3 ver
    for the benefit of the user (the CCA3 does not have this information) --
    set up position of the lost hiker, the CCA3
    will be undergoing active development to allow automatic activation of a variety
    of environments, not just forest one
    '''
    print("for benefit of user (CCA3 does not get this information):")
    print("Let's put the CCA1 in the corner at m,n = (1,1): ")
    d.set_cca1(1, 1)
    print("\nNow let's put the lost hiker at  position m,n = (4,2): ")
    print("The CCA1, of course, does not have access to hiker position")
    d.set_hiker(4, 2)
    set_goal_and_hippo(d, g, h)
    return True


def autonomic_check(g) -> bool:
    '''in_use_do_not_archive
    CCA3 ver
    central autonomic module is checked for priority operations
    before external sensory input vectors are processed
    note: peripheral autonomic modules cannot be checked until
    the first stages of external sensory input vectors in the
    evaluation cycle
    '''
    g.large_letts_display("AUTONOMIC MODULE SIMULATION")
    g.fast_input("\nPress ENTER to continue...\n")
    autonomic_sleep_wake(g)
    print(colored("Autonomic system was not modeled in the equations of the CCA3 Binding paper", 'cyan'))
    print("Core CCA3 autonomic check is passed -- no set of immediate actions required.")
    print("No attention needed for any CCA3 peripheral autonomic actions.\n")
    return True


def autonomic_sleep_wake(g) -> bool:
    '''in_use_do_not_archive
    CCA3 ver
    sleep/wake and energy cycles
    '''
    if not g.fastrun:
        print("\nSimplified simulation of sleep/wake cycle and energy managment.")
        print("CCA3 in wake state. Energy usage state is normal.")
    return True


def devpt_timer(d, g):
    '''depending on the age of the CCA1 different procedural vectors are
    returned from the instinctual core goals module
    -the instinctual core goals module, which is affected by the maturity stage of the CCA1 via
     this developmental timer, feeds intuitive logic, intuitive physics,
     intuitive psychology and/or intuitive goals planning procedural vectors into the groups of
     HLNs configured as logic/working memory units.
    nano version -- simulation to allow drop insertion of more authentic
     components
    '''
    all_goals_dict = {
        g.goal_random_walk: "random walk",
        g.goal_skewed_walk: "skewed to S and E given 0 0 start(==navgn help)",
        g.goal_precausal_find_hiker: "find the hiker with associative and some precausal behavior",
        g.goal_causal_find_hiker: "find the hiker with associative, precausal and causal behavior",
    }
    if d.age_autonomic_calls < 10000:
        return all_goals_dict
    all_goals_dict["GOAL_POST_MATURE"] = "consider transfer of learned knowledge"
    return all_goals_dict


def set_goal_and_hippo(d, g, h, goal_desired: str = "") -> None:
    '''at start of program the current goal of the instinctual core goals module must
    be set, for example to find a lost hiker or perhaps just to display a random walk
    -this is determined by setting here: h.current_hippocampus, d.current_goal
    -indirectly also via setting here: h.meaningfulness
    -set in other methods: d.current_autonomic, d.current_instinct
     (via get_current_autonomic(), get_current_instinct() )
    -allowed_goals = all_goals_dict returned from calling devpt_timer:
        GOAL_RANDOM_WALK = '00000000'U
        GOAL_SKEWED_WALK = '00000001'
        GOAL_PRECAUSAL_FIND_HIKER = '11111111'
        GOAL_CAUSAL_FIND_HIKER = '11110000'
        DEFAULT_GOAL = GOAL_RANDOM_WALK
    -allowed hippocampus settings:
        if current_goal == g.goal.random_walk, g.goal_skewed_walk:
            h.current_hippocampus = 'LAMPREY'
        if current_goal == g.goal_precausal_find_hiker:
            h.current_hippocampus = 'REPTILE'
        if current_goal == g.goal_causal_find_hiker:
            h.meaningfulness = True
            h.current_hippocampus = 'HUMAN'
    -h.meaningfulness default value is False, but note that if d.current_goal is
    g.goal_causal_find_hiker then h.meaningfulness is set to True
    -these settings are used in various ways to affect the behavior and performance
    of different models of the CCA1 (ie, g.current_hippocampus, d.current_goal,
    h.meaningfulness and also but set elsewhere d.current_autonomic and
    d.current_instinct)

    parameters:
        d, g -- standard instantiations of ddata.py classes which hold data
        goal_desired -- if pass in the goal desired it will used to then set
          d.hippocampus and h.meaningfulness
    output:
        None  (settings done directly to d and g instances)
    '''
    # automatic settings if g.fastrun flag is true
    if g.fastrun:
        print("debug: g.fastrun flag is true, thus automatically set:")
        print("    d.current_goal = g.goal_causal_find_hiker")
        print("    h.current_hippocampus = HUMAN")
        print("    h.meaningfulness = True\n")
        d.current_goal = g.goal_causal_find_hiker
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True
        return

    # developmental timer affects which goals are made available to the CCA1
    # allowed_goals =returned all_goals_dict -- values shown above
    allowed_goals = devpt_timer(d, g)

    # if specify the chosen goal then it is used to set hippocampus and
    # meaningfulness as well putting chosen goal into d.current_goal
    # leave if..else expanded out so can easily fine-tune specs in future
    if goal_desired in allowed_goals:
        print("goal_desired parameter is: ", goal_desired)
        print("thus this become d.current_goal, and h.current_hippocampus")
        print("and h.meaningfulness will be set accordingly\n")
        if goal_desired == g.goal_random_walk:
            d.current_goal = goal_desired
            h.current_hippocampus = "LAMPREY"
            h.meaningfulness = False
            return
        if goal_desired == g.goal_skewed_walk:
            d.current_goal = goal_desired
            h.current_hippocampus = "LAMPREY"
            h.meaningfulness = False
            return
        if goal_desired == g.goal_precausal_find_hiker:
            d.current_goal = goal_desired
            h.current_hippocampus = "REPTILE"
            h.meaningfulness = False
            return
        if goal_desired == g.goal_causal_find_hiker:
            d.current_goal = goal_desired
            h.current_hippocampus = "HUMAN"
            h.meaningfulness = True
            return
        # default to HUMAN settings if specified goal is not listed above
        d.current_goal = g.goal_causal_find_hiker
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True
        return
    if goal_desired != "" and goal_desired not in allowed_goals:
        # default to HUMAN settings if specified goal is not a valid one
        print("debug: goal_desired parameter is not in allowed_goals, thus:")
        print("    d.current_goal = g.goal_causal_find_hiker")
        print("    h.current_hippocampus = HUMAN")
        print("    h.meaningfulness = True\n")
        d.current_goal = g.goal_causal_find_hiker
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True
        return

    # if g.fastrun or goal_desired not specified, user will now have to choose
    # the type of hippocampus which will affect settings then of
    # d.current_goal, h.meaningfulness
    print('\nPlease choose type of "hippocampus"/"brain" which, of course,')
    print(" only loosely approximates the biological equivalent:")
    print("1. Lamprey hippocampal/brain analogue")
    print("2. Fish hippocampal/telencephalon analogue")
    print("3. Reptile hippocampal/pallium analogue")
    print("4. Mammalian hippocampus - note: meaningfulness, precausal")
    print("5. Human hippocampus - note: meaningfulness plus full causal features")
    print("6. Augmented Human level 1 - simultaneous multiple navigational threads")
    print("7. Augmented Human level 2 - algorithm center in each navigational module")
    try:
        b_b = int(input("Please make a selection:"))
    except:
        print(
            "ENTER or nonstandard input, thus defaults to the causal human hippocampus selection"
        )
        b_b = 5
    if b_b not in range(1, 8):
        print("Default causal human hippocampus selected")
        b_b = 5
    if b_b == 1:
        print("\nWill default at this time to a quasi-skewed walk")
        print(
            "Current status is clean fcnl simulation to allow drop in of more authentic and sophisticated"
        )
        print(" components in finer-grain simulation level.\n")
        h.current_hippocampus = "LAMPREY"
        h.meaningfulness = False
        # will default to quasi-skewed walk
    if b_b == 2:
        # h.current_hippocampus = 'FISH'
        print("\nWill revert at this time to lamprey pallium analogue")
        print(
            "Important evolutionary and conceptual advances to be put in coming versions"
        )
        print(" of nano simulation and allow selection of fish pallium")
        print("Note that fish brain does not allow meaningfulness.\n")
        h.current_hippocampus = "LAMPREY"
        h.meaningfulness = False
    if b_b == 3:
        # some precausal features'
        print(
            "\nWill default at this time to simple pallium analogue with some precausal features"
        )
        print(
            "Current status is clean functional simulation to allow drop in of more authentic and sophisticated"
        )
        print(" components in finer-grain simulation level")
        print("Note that reptilian brain does not allow meaningfulness.\n")
        h.current_hippocampus = "REPTILE"
        h.meaningfulness = False
    if b_b == 4:
        print("\nWill revert at this time to reptile pallium analogue")
        print(
            "Important evolutionary and conceptual advances to be put in coming versions"
        )
        print(
            " of nano simulation and allow selection of mammalian hippocampus and brain"
        )
        print("However, given mammalian brain, meaningfulness is present.\n")
        h.current_hippocampus = "REPTILE"
        h.meaningfulness = True
    if b_b == 5:
        # pending status, some simple causal features
        print(
            "\nWill default at this time to  a brain with associative, precausal and some genuine"
        )
        print(" causal features. Given mammalian brain, meaningfulness is present.\n")
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True
    if b_b == 6:
        # h.current_hippocampus = 'SUPERINTELLIGENCE'
        print(
            "\nWill default at this time to a simplified human brain with some associative,"
        )
        print(
            "  precausal and some genuine causal features. Neither the coarse grain functional simulation nor"
        )
        print(
            "  the finer grain simulations have been set up for superintelligence abilities,"
        )
        print(
            "  but the possibility exists of implementing in the future by simple modifications"
        )
        print("  of human hippocampus and overall CCA1 structure and function")
        print("Given supra-mammalian brain, meaningfulness is present.\n")
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True
    if b_b == 7:
        # h.current_hippocampus = 'SUPERINTELLIGENCE2'
        print(
            "\nWill default at this time to a simplified human brain with some associative,"
        )
        print(
            "  precausal and some genuine causal features. Neither the coarse grain functional simulation nor"
        )
        print(
            "  the finer grain simulations have been set up for superintelligence abilities,"
        )
        print(
            "  but the possibility exists of implementing in the future by simple modifications"
        )
        print("  of human hippocampus and overall CCA1 structure and function.")
        print(
            "Superintelligence2 refers to more sophisticated modifications to CCA1 structure than"
        )
        print("  in the Superintelligence 1 design.")
        print("Given supra-mammalian brain, meaningfulness is present.\n")
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True

    # now set d.current_goal of finding hiker from settings of h.current_hippocampus
    if h.current_hippocampus == "LAMPREY":
        d.current_goal = g.goal_skewed_walk
    elif h.current_hippocampus == "REPTILE":
        d.current_goal = g.goal_precausal_find_hiker
    elif h.current_hippocampus == "HUMAN":
        d.current_goal = g.goal_causal_find_hiker
    else:
        print("debug: set_goal_and_hipp() d.current_goal not properly set")
        print("debug: will set to human settings to avoid runtime issues")
        d.current_goal = g.goal_causal_find_hiker
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True
    if g.debug:
        print("\nd.current_goal is set to {}".format(d.current_goal))
        print("h.current_hippocampus is set to {}".format(h.current_hippocampus))
        print("Meaningfulness status: ", h.meaningfulness)
    return


def beep_secs(secs: int = 1, force_user_key_entry_after_beep: bool = True) -> bool:
    '''beeps for time above
    parameters: secs: int -- how long the beep should last
                             if -1 then bypasses the routine
    returns:
        True: bool -- if sound or visual input got attention of user (or routine bypassed)
                      (assume sound works since slows program inputs for 1 second)
        False: bool -- visual input did not get attention of user
    current implementation may or may not beep if non-windows platform
    '''
    if secs == -1:
        print("\n\nbeeping bypassed\n\n")
        return True
    try:
        # pylint: disable=import-outside-toplevel
        import winsound  # type: ignore

        duration = secs * 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
        if force_user_key_entry_after_beep:
            # clear input buffer -- windows platform
            while msvcrt.kbhit():  # type: ignore
                msvcrt.getch()  # type: ignore
            # allow user to look at screen associated with beep
            input("Please press any key to continue")
        return True
        # pylint: enable=import-outside-toplevel
    except:
        print("Windows beeping sound via <import winsound> (also msvcrt used) did not work on your platform")
        print("\n\n**VISUAL BEEP**    **LOOK AT SCREEN**\n\n")
        if force_user_key_entry_after_beep:
            input("Please press any key to continue")
            return True
        return False


def checkpoint_beep(secs=1, to_print="checkpoint_beep....", arrows=True, ex=""):
    '''useful for devp't work to avoid going into the slow debugger where want
    to run through hundreds of lines before a condition may
    beep will occur followed by need to press ENTER
    '''
    beep_secs(secs)
    if arrows:
        print("---------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if ex != "":
        print("value of variable selected to examine is: ", ex)
    input(to_print)
    return True


def input_vectors_shaping(d, g, sense_inputs: list, to_print: str) -> list:
    '''get raw input from sensors via hardware or manually or emulation
    calls get_emulated_input(sensory_possibles) for the NESW emulation
    sense_inputs is the list of the possible sensory inputs
    return slightly processed input as NESW list of strings
    parameters--
        d, g - project standard instantiations of ddata.py classes
        to_print - message to print when this method starts
        sense_inputs - eg, d.visual_database, d.auditory_database, etc
            the matrix with emulation of possible inputs of a given sense
            for a given square in the N,E,S, and W directions
    output--
        returns a list [N,E,S,W] containing 4 8-bit values, with each of
            these values corresponding to a sensory (for sense x) input
            for the CCA1's current position in the N, E, S and W direction
    '''
    if to_print == "":
        to_print = "SPECIFY SUBMODULE OR MODULE NAME"
    print("\n-->", to_print)
    # get input from sensors
    if g.fastrun:
        aa = get_emulated_input(d, g, sense_inputs)
    elif g.hardware:
        aa = get_hardware_input(d, g, "")
    else:
        aa = input(
            "Press ENTER for automatic emulation of North-E-S-W sensors, or m to enter manually:"
        )
        if aa == "":
            aa = get_emulated_input(d, g, sense_inputs)
        else:
            aa = [0, 0, 0, 0]
            aa[0] = input(
                "Enter NORTH direction sensory input (1 and 0s): (nb -1 N triggers demo case)"
            )
            aa[1] = input("Enter EAST direction sensory input (1 and 0s): ")
            aa[2] = input("Enter SOUTH direction sensory input (1 and 0s): ")
            aa[3] = input("Enter WEST direction sensory input (1 and 0s): ")
            # trigger causal feedback of demo case
            if aa[0] == "-1":
                print("entered -1 and will cause causal demo case 00000000 in all NESW")
                aa[0] = "00000000"
                aa[1] = "00000000"
                aa[2] = "00000000"
                aa[3] = "00000000"
    # process raw input
    for i in range(4):
        aa[i] = str(aa[i])
        aa[i] = aa[i] + g.filler
        aa[i] = aa[i][0:8]
    print(
        "cca1 position: ", d.cca1_position, " and input_vectors_shaping value of:\n", aa
    )
    return aa


def simulated_vision_to_vector(x: int, y: int, direction: int, g, h) -> str:
    '''CCA3 ver
    -in CCA3 re-write please create a separate method for each sensory system
    -while, for example, there are many similarities in creating a simulated olfactory
     or tactile or auditory or radar sensory input to vector method, each sensory modaility
     involves specialized signal processing, and this is respected in the method created

    -simulates a physical video camera looking N, E,S, or W at a specifed 'GPS' location
    (which is simply an x,y location in the simulated forest world or gear world, etc)
    -simulates the raw video signal, video pre-processing and processed signal output
    -CNN object detection, RNN object time series detection as well as the more powerful (we
     believe) map-based neural network dectection are handled in other methods which call
     this method (the vector output of those downstream methods, is then sent to the
     object segmentation gateway module of the navigation module)

    -assume 6x6 grid map of the forest in one of the navigation maps
    -assume edge squares which cannot be used for movement, thus possible
    squares for CCA3 are 1,1 (square 0) --> 4,4 (square 15)
    -in future, really should consider getting rid of edge squares and allowing maps
    of infinite size, but ok now for toy examples
    -0,0 or 1,0 for example would be edge squares not allowed by this method
    -direction is an integer 0,1,2, or 3 corresponding to N,E,S,W

    input parameters--
        x, y - coordinates on the navigation map corresponding to the matrix of
            simulated visual inputs
        direction -- the CCA3 is in the x,y specified square and looking in this specified
            direction for this sensory input
            currently due to limitations of the sensory matrices simple discrete values
               0,1,2, or 3 which corresponds to N,E,S,W
        g, h - project standard instantiations of data classes
               h.forest_visual_sensory_inputs_simulation holds the simulated possibe visual sensory inputs

    output--
        returns simulated visual input in specified direction (8 bit vector (string)obtained from h.forest_visual_sensory_inputs_simulation)

    TOTAL_ROWS = 6  #count EDGE squares
    TOTAL_COLS = 6  #count EDGE squares
    pass-----------------
    '''
    # input simulated signal processing of visual signal
    if h.vis_forest.size != 16 * 4:
        print(
            "\n****debug: there does not appear to be data for all squares 0->15 in 4 NESW directions"
        )
        print(
            "Further and routine visual signal processing pending future versions...."
        )
        return "-1"

    # check validity of position, do not allow to be in edge squares
    # assume edge squares in the physical world for the forest-lost hiker scenario
    if x <= 0 or y <= 0 or (x >= g.total_rows - 1) or (y >= g.total_cols - 1):
        print(
            f"\n****debug: x,y square coordinates {x},{y} invalid -- simulated_vision_to_vector()"
        )
        return "-1"

    # what are the adjacent square possible sensory input values in the N or E or S or W direction?
    # convert x,y into square 0-15 represented in the matrix of possible simulated visual sensory inputs
    # for every x,y square = [[N possible inputs] [E possible inputs] [S possible inputs] [W possible inputs] ]
    # x,y parameter coordinates give square_number, and direction parameter gives direction
    square_number = 4 * (x - 1) + (
        y - 1
    )  # squares 0 (1,1) - 15 (4,4) in the 6x6 grid (outer squares are edge squares)
    possibles = h.vis_forest[square_number][
        direction
    ]  # gives possible sensory inputs for that square looking in the specified direction
    # print(possibles) #eg, ['191111100', '119111101', '119011000', '100911100']
    # ensure that there is at least one sensory value to choose from in that specified direction
    if len(possibles) == 0:
        print(
            "\n****debug: no input sensory value in direction {direction} for {x}, {y} -- simulated_vision_to_vector()"
        )
        print("****debug: thus following value will be used: ", g.filler)
        simulated_visual_input_this_direction = [g.filler]
        print(
            simulated_visual_input_this_direction
        )  # should be g.filler which is '00000000'
    else:
        simulated_visual_input_this_direction = possibles[
            random.randint(0, len(possibles) - 1)
        ]  # randomly choose one of these sensory values
        # print(simulated_visual_input_this_direction) #eg, randomly selects '119111101'

    # return simulated visual input in specified direction
    # print('the type of simulated_visual_input_this_direction is actually: ', type(simulated_visual_input_this_direction))
    return simulated_visual_input_this_direction  # type:ignore


def associative_match_visual_input(d, sensory_input_one_direction: str) -> tuple:
    '''CCA3 ver
    -simulated_vision_to_vector() produced a simulated vision signal for specified
      square and direction
    -this method uses an associative matching algorithm to match the signal to closest
      labeled signal in the dictionary ddata.self.visual_dict
    -a shallow ml algorithm, a deep algo or in fact a simply fuzzy logic algorithm is
      fine for this matching since it no longer forms the main basis for binding
      sensory information in the CCA3
    -binding is now done by matching not a label of a various sense, but rather the
      features of that sensory signal onto a navigation map
    -thus, this associative_match method provides a useful label for the signal that
      can be used for other things, but it is no longer the main means of binding
      sensory information into the representations within the CCA3

    -currently fuzzy associative matching library is used:
            from fuzzywuzzy import fuzz
            from fuzzywuzzy import process
            extract' method imported from fuzzywuzzy library

    input parameters:
    -sensory_input_one_direction which is the simulated vision signal, ie,
      simulated_visual_input_this_direction from simulated_vision_to_vector() method
    -d ->
    -d.visual_dict (called directly within the method)
    {'11100011': 'lake', '01010000': 'lost hiker visual', '11111100': 'obstruction',
    '00010001': 'shallow river', '00011001': 'shallow river + spraying water',
    '11000110': 'forest'}

    output:
    -returns the most likely label for that vision signal referencing d.visual_dict and exact
      vector matched which is the key for that label
    '''
    # print('sensory_input_one_direction: ', sensory_input_one_direction)
    # print('d.visual_dict: ', d.visual_dict, '\n')
    # print('d.visual_dict.keys(): ', d.visual_dict.keys())
    top_matched_forest_keys_with_rating = process.extract(
        sensory_input_one_direction, d.visual_dict.keys(), limit=1
    )
    # print('top_matched_forest_keys_with_rating: ', top_matched_forest_keys_with_rating)
    top_key = [top_matched_forest_keys_with_rating][0][0][0]
    label = d.visual_dict[top_key]
    # print(label, top_key)
    return label, top_key


def generate_features_for_visual_input(d, h, matched_sensory_vector: str):
    '''CCA3 ver
    -simulated_visual_to_vector() produced a simulated visual signal for specified
      square and direction
    -associative_match_visual_input() uses an associative matching algorithm to match the signal
      to closest labeled signal in the dictionary ddata.self.visual_dict
    -however, in the CCA3 we no longer bind a label but bind the real world features of a sensory
      input to a navigation map
    -since sensory inputs are being simulated (as opposed to using real video,audio, tactile, etc sensors)
      we need to generate, ie, simulate, the real world features for our visual signal label
    -this binding information can then be used to bind the sensory input onto a navigation map
    -at present, we simply pull these features from a pre-constructed d.aud_features dictionary

    -note that this method takes a single sensory vector and generate a full map of
      what a real world set of sensors would pick up *before* the label for that single sensory vector
      was generated
    -i.e., this method is *generating* features to simulate what sensors would be sensing and
      representing on a navigation map
    -this navigation map is not a map of the forest or whatever simulation environment is being run,
      but rather a map of the sensory scene in front of, ie, in the direction the CCA3 is sensing
    -in real world physical embodiment what would happen is that visual and audio and other sensors would
      sense features of the sensory scene in front of the CCA3 and these would be represented on a
      navigation map (h.gb datastructure)
    -also of importance, this navigation map (h.gb datastructure) is rarely de novo but rather a_precombos
      navigation map that matches most closely to the sensory scene is retrieved from the causal memory
      and then modified or updated by the features of the sensory scene sensed

    -there is introduction of noise into the raw_feature_list since this is supposed to be a simulation of the
      input sensors sensing the sensory scene in front of the CCA3 which is then matched with a stored
      navigation map in the next method

    input parameters:
    -d, h -- data and method structures instantiations
    -matched_sensory_vector is the matched visual input vector (ie, matched with dictionary above)
    -d.vis_features (called directly within the method)
     see below for detailed description

    output:
    -returns h since this method can modify h
    (the features for this sensory signal and where on the 6x6 navigation map they belong
     [feature1, x1, y1, feature2, x2, y2, .....] via h.gb with h data structure)


    d.vis_features is a dictionary --
      -key==matched_sensory_vector ie, a label of what the sense determined the object to be
      -value==list of features to be synthetically generated, ie, simulated, for this object
         since we need to simulate various features the sensors would sense and then map (ie, bind)
         these features onto a navigation map
      -structure of the value term:  [ [list of a feature at coordinate], [another list], ... [another list]
      thus,  key:[ [list of a feature], [list of a feature].....[list of a feature]]
      (and thus vis_features = { key:[[lists]], key:[[lists]].....key:[[lists]]}
      -structure of a list of the value term:
          eg,  ['visual', 'water', 1, 3, 1]
               [sensory system, basic feature, intensity of feature, x, y]
               thus the visual system is reporting at (3,1) water of intensity 1
               thus water of intensity 1 will be mapped in the (3,1) grid of nav map
     -special basic features:
        'label' -- this refers to a text or verbal label, it is not mapped onto any
           one location of the nav map but refers to the entire nav map x,y==0,0
           'label' is a sensory system just like 'visual' or 'olfactory' or 'audio'
        'feature label' -- not incorporated at present, perhaps in the future, however,
           symbol grounding in the CCA3 is by virtue of links to other nav maps, not labels
        'link' -- the x,y give coordinates in the next higher map, the intensity value gives
          intensity of the link, a sensory system can be specified, eg, 'visual', 'auditory', etc
          future version more sophisticated multidimensional coordinate system
          eg, ['visual', 'link', 1, 0, 0] -- set up for link but not used at present,
              not going anywhere at present
         <basic features> -- eg, 'water' actually will automatically link to more
              basic maps of their properties, but few 100 basic features which can be
              used without overt links

    h.gb = "gb" == "go_board" == "navigation map which main processing is done on"
    -in 2D we would use: 6x6x40, ie, 6x6 x,y grid where each grid square can have 40 features mapped
        we need to add dimension 'c' to count the different features, ie, x,y,c where c 0-->39
    -in 3D we would use: 6x6x6x40, ie, 6x6x6 x,y,z cube where each cube can have 40 features mapped
       (i.e., x,y,z,c specified for each feature where c is feature number 0-->39 )(we can specify 40 different features)
    -note that 3 dimensional navigation map is supported by biological research -- hippocampus place
        fields may very well be 3D, eg, Nachum Ulanovsky's work with hippocampal spatial representations
        in the echolating bats
    -in 3D + time we would use for 6 time intervals: 6x6x6x6x40 ie, 6x6x6x6 x,y,z,t hypercube (tesseract if 4D)
        each with up to 50 different features mapped
        (i.e., x,y,z,t,c specified for each feature where c is feature number 0-->49 )(we can specify 50 different features
        for each x,y,z,t,n hypercube)
    -we use a navigation map as the main navigation map in the navigation module, but we also have hundreds if
        not millions or billions of navigation maps saved to (and retrieved from) the causal memory module,
        thus need to specify which navigation map "n" a feature is in; if we use n=100 navigation maps for our
        toy example, thus we would use a structure with these dimensions: 6x6x6x6x100x40
        (i.e., x,y,z,t,n specified for each feature where we allow (x,y,z,t,n) x40 different features)
        (i.e., 5D-hypercube or 5-cube or penteract where we allow 50 5-cubes to allow 50 different features mapped 0--49)
        h.gb[x,y,z,t,n,c] can specify for a feature, i.e.,
        h.gb[x,y,z,t,n,c] = feature f
        (6,6,6,6,100,50) total dimensions
    -the "feature" or "f" structure is specified above but we don't need the coordinates in feature any longer
        since the coordinate are specified by our data structure as well as which feature it is 0-->49
        thus the structure of the feature or f:
        f = [sensory system 0, basic feature 1, intensity/color/modulator of feature 2]

    -features are now mapped topologically by sensory sytem:
            0 -   9 labels
            10 - 19 links
            20 - 29 visual
            30 - 35 audio
            36 - 39 olfactory
            40 - 45 other1
            46 - 49 other2
       -eg, above the feature  ['visual', 'water', 1, 3, 1] if it was stored in our data structure in the
       the main navigation map n=0 and at time t=0 and using 2D thus z=0
       -'visual' thus topologically must be mapped c=20...29 possible locations
       -let's assume this is the second *visual* feature we are mapping to that particular hypercube, thus previous
       feature counter number was c=20, and thus this one will be feature counter number c=21
       h.gb[x,y,z,t,n,c] = h.gb[3,1,0,0,0,21] = f = ['visual', 'water', 1]


    -NavMod() class:  self.gb = np.empty((6,6,6,6,100,50), dtype=object)
        (100 navigation maps for our toy example; we are allowing 50 different features to be mapped to each hypercube of
         this data structure 'gb' which constitutes the causal memory of the CCA3)
        (of interest, thus 6.5 million features in total can be specified for these 100 navigation maps our
        toy CCA3 is composed of; however, the CCA3's causal memory is more of a sparse model than the dense models
        of traditional ANNs, thus assume no more than a twentieth of that, ie, about 250K features used at maximum,
        and if each feature is 100 bytes, then about 25MB for data structure)
        (this represents 25MB/100 maps  = 250KB per map)
        (if scale to human 300M cortical columns thus analogous 300M navigation maps, 300M*250KB ~ 1 TB)

    '''
    # for devpt purposes -- put back into ddata.py, and matched_sensory_vector should be parameter
    vis_features = {
        "00010001": [
            ["label", "shallow river", 1, 0, 0],
            ["visual", "link", 1, 0, 0],
            ["visual", " water", 1, 3, 0],
            ["visual", "water", 1, 3, 1],
            ["visual", "water", 1, 4, 2],
            ["visual", "water", 1, 2, 3],
            ["visual", "water", 1, 2, 4],
            ["visual", "water", 1, 1, 5],
            ["auditory", "bubbling", 1, 1, 2],
            ["auditory", "bubbling", 1, 4, 3],
            ["olfactory", "musty", 1, 0, 4],
            ["**", "3,4", 1, 3, 4],
            ["**", "1,3", 1, 1, 3],
        ],
        "11100011": [["lake"]],
        "01010000": [["lost hiker visual"]],
        "11111100": [["obstruction"]],
        "00011001": [["shallow river + spraying water"]],
        "11000110": [["forest"]],
        "11000000": [["forest noise visual feature"]],
    }
    matched_sensory_vector = "00010001"
    d.pass_d()

    # obtain raw_feature_list corresponding to the identified scene
    try:
        raw_feature_list = vis_features[matched_sensory_vector]
    except:
        print(
            "\n****debug: in generate_features_for_visual_input(): no match in d.vis_features"
        )
        return h
    # print('matched_sensory_vector: ', matched_sensory_vector, '\n')
    # print('vis_features: ', vis_features, '\n') #{'00010001':[['label', 'shallow river',1, 0 ,0,], ['visual', 'link', 1, 0, 0],.....
    # print('raw_feature_list ', raw_feature_list, '\n') #[['label', 'shallow river',1, 0 ,0,], ['visual', 'link', 1, 0, 0],.....
    # print('raw_feature_list[0] ', raw_feature_list[0], '\n') #['label', 'shallow river',1, 0 ,0,]
    # print(len(raw_feature_list[0])) #5

    # introduction of noise into the raw_feature_list since this is supposed to be a simulation of the input sensors sensing the
    # sensory scene in front of the CCA3 which is then matched with a stored navigation map
    if len(raw_feature_list) == 0:  # type:ignore
        print(
            "\ndebug: there are no features to map -- generate_features_for_visual_input()"
        )
        return h
    if len(raw_feature_list) > 2:  # type:ignore
        # if 3 or more features then will delete one to introduce less than perfect map
        raw_feature_list.pop(random.randint(0, len(raw_feature_list) - 1))  # type:ignore
        print("noisy raw feature list: ", raw_feature_list, "\n")  # for debug

    # now go through sensed features and map to h.gb navigation map structure
    # eg, h.gb[x,y,z,t,n,c] = h.gb[3,1,0,0,0,1] = f = ['visual', 'water', 1]
    for feature in raw_feature_list:  # type:ignore
        # data integrity and very simple cleanup algorithm
        if len(feature) != 5:
            print("\ndebug: feature retrieved: ", feature)
            print(
                "\ndebug: simple data cleanup for feature -- generate_features_for_visual_input()\n"
            )
            f0 = "visual"  # f0 = feature[0]
            f1 = "null"  # f1 = feature[1]
            f2 = 1  # f2 = feature[2]
            x = 0  # x =  feature[3]
            y = 0  # y =  feature[4]
        else:
            # otherwise read in features from feature iterated object
            # stucture: [sensory system 0, basic feature 1, intensity of feature 2, x 3, y 4]
            f0 = feature[0]
            f1 = feature[1]
            f2 = feature[2]
            x = feature[3]
            y = feature[4]

        # [f0,f1,f2] is new feature 'f' we want to map to h.gb[x,y,z,t,n,c]
        z = 0  # currently only using 2D sensory world
        t = 0  # currently not using time series or intervals
        n = 0  # main navigation map is map 0
        # print(f0,f1,f2,x,y,z,t,n) #for debug
        # print('feature in raw_feature_list is: ', feature)  #for debugger

        # c is current feature number being mapped for each coordinate hypercube
        # 0 -   9 label, 10 - 19 visual, 20 - 29 auditory, 30 - 34 olfactory,
        # 35 - 39 tactile, 40 - 44 other1, 45 - 49 other2
        if f0 == "label":
            start_feature = 0
            stop_feature = 9
        elif f0 == "visual":
            start_feature = 10
            stop_feature = 19
        elif f0 == "auditory":
            start_feature = 20
            stop_feature = 29
        elif f0 == "olfactory":
            start_feature = 30
            stop_feature = 34
        elif f0 == "tactile":
            start_feature = 35
            stop_feature = 39
        elif f0 == "other1":
            start_feature = 40
            stop_feature = 44
        elif f0 == "other2":
            start_feature = 45
            stop_feature = 49
        else:
            print("\ndebug: sensory system not recognized, mapped as other2")
            start_feature = 45
            stop_feature = 49

        for c in range(
            start_feature, stop_feature + 2
        ):  # eg, tactile 35->39+2 thus 35..40 in loop
            if c > stop_feature:  # eg, >39
                # print('\ndebug: no empty feature slots thus oldest one will be overwritten')
                # print(f'gb[{x,y,z,t,n,start_feature}] = [{f0,f1,f2}]')
                h.gb[x, y, z, t, n, start_feature] = [f0, f1, f2]
                break

            if h.gb[x, y, z, t, n, c] is None:
                # print(f'c={c} is empty and thus will be used')
                # print(f'gb[{x,y,z,t,n,c}] = [{f0,f1,f2}]')
                h.gb[x, y, z, t, n, c] = [f0, f1, f2]
                break
        # input('feature mapped to h.gb.... press to continue\n') # for debug
    print("\nall features mapped to gb-0 navigation map\n\n")
    return h


def simple_visualize_gb(h):
    '''CCA3 ver
    x-y plane quick visualization of h.gb
    accumulate all the c value planes in this simple visualization
    h.gb[x,y,z,t,n,c] = [f0,f1,f2]
    '''
    # create x-y plan xy_plane
    xy_plane = np.empty((6, 6), dtype=object)
    for x in range(6):
        for y in range(6):
            feature_accumulator = []
            for c in range(50):
                bb = h.gb[x, y, 0, 0, 0, c]
                if bb is not None:
                    feature_accumulator.append(bb)
            xy_plane[x, y] = feature_accumulator

    # print out xy_plane
    print("\nSimplified Visualization of h.gb(n=0)  NavMap#0")
    print("Show all Features (c=0..49)")
    print("x (or m) across 0->5, y (or n) down 0->5")
    print("----------------------------------------\n")
    for y in range(6):
        print(f"{y}", end=":  ")
        for x in range(6):
            print(xy_plane[x, y], end=f"  <-{x}-  ")
        print("\n\n")
    return True


def simulated_auditory_to_vector(x: int, y: int, direction: int, g, h):
    '''CCA3 ver
    -in CCA3 re-write please create a separate method for each sensory system
    -while, for example, there are many similarities in creating a simulated olfactory
     or tactile or auditory or radar sensory input to vector method, each sensory modaility
     involves specialized signal processing, and this is respected in the method created

    -simulates a physical auditory sensor listening N, E,S, or W at a specifed 'GPS' location
    (which is simply an x,y location in the simulated forest world or gear world, etc)
    -simulates the raw auditory signal, auditory pre-processing and processed signal output
    -CNN object detection, RNN object time series detection as well as the more powerful (we
     believe) map-based neural network dectection are handled in other methods which call
     this method (the vector output of those downstream methods, is then sent to the
     object segmentation gateway module of the navigation module)

    -assume 6x6 grid map of the forest in one of the navigation maps
    -assume edge squares which cannot be used for movement, thus possible
    squares for CCA3 are 1,1 (square 0) --> 4,4 (square 15)
    -in future, really should consider getting rid of edge squares and allowing maps
    of infinite size, but ok now for toy examples
    -0,0 or 1,0 for example would be edge squares not allowed by this method
    -direction is an integer 0,1,2, or 3 corresponding to N,E,S,W

    input parameters--
        x, y - coordinates on the navigation map corresponding to the matrix of
            simulated auditory inputs
        direction -- the CCA3 is in the x,y specified square and 'looking' (actually listening) in
            this specified direction for this sensory input
            currently due to limitations of the sensory matrices simple discrete values
               0,1,2, or 3 which corresponds to N,E,S,W
        g, h - project standard instantiations of data classes
               h.forest_auditory_sensory_inputs_simulation holds the simulated possibe visual sensory inputs

    output--
        returns simulated auditory input in specified direction (8 bit vector (string)obtained from h.forest_visual_sensory_inputs_simulation)

    TOTAL_ROWS = 6  #count EDGE squares
    TOTAL_COLS = 6  #count EDGE squares
    pass-----------------
    '''
    # input simulated signal processing of auditory signal
    print("h.aud_forest.size ", h.aud_forest.shape, h.vis_forest.shape)
    if h.aud_forest.size != 16 * 4:
        print(
            "\n****debug: there does not appear to be data for all squares 0->15 in 4 NESW directions"
        )
        print(
            "Further and routine auditory signal processing pending future versions...."
        )
        return "-1"

    # check validity of position, do not allow to be in edge squares
    # assume edge squares in the physical world for the forest-lost hiker scenario
    if x <= 0 or y <= 0 or (x >= g.total_rows - 1) or (y >= g.total_cols - 1):
        print(
            f"\n****debug: x,y square coordinates {x},{y} invalid -- simulated_auditory_to_vector()"
        )
        return "-1"

    # what are the adjacent square possible sensory input values in the N or E or S or W direction?
    # convert x,y into square 0-15 represented in the matrix of possible simulated visual sensory inputs
    # for every x,y square = [[N possible inputs] [E possible inputs] [S possible inputs] [W possible inputs] ]
    # x,y parameter coordinates give square_number, and direction parameter gives direction
    square_number = 4 * (x - 1) + (
        y - 1
    )  # squares 0 (1,1) - 15 (4,4) in the 6x6 grid (outer squares are edge squares)
    possibles = h.aud_forest[square_number][
        direction
    ]  # gives possible sensory inputs for that square looking in the specified direction
    # print(possibles) #eg, ['191111100', '119111101', '119011000', '100911100']
    # ensure that there is at least one sensory value to choose from in that specified direction
    if len(possibles) == 0:
        print(
            "\n****debug: no input sensory value in direction {direction} for {x}, {y} -- simulated_auditory_to_vector()"
        )
        print("****debug: thus following value will be used: ", g.filler)
        simulated_auditory_input_this_direction = [g.filler]
        print(
            simulated_auditory_input_this_direction
        )  # should be g.filler which is '00000000'
    else:
        simulated_auditory_input_this_direction = possibles[
            random.randint(0, len(possibles) - 1)
        ]  # randomly choose one of these sensory values
        # print(simulated_auditory_input_this_direction) #eg, randomly selects '119111101'

    # return simulated auditory input in specified direction
    # print('the type of simulated_auditory_input_this_direction is actually: ', type(simulated_auditory_input_this_direction))
    return simulated_auditory_input_this_direction  # type:ignore


def associative_match_auditory_input(d, sensory_input_one_direction: str):
    '''CCA3 ver
    -simulated_auditory_to_vector() produced a simulated auditory signal for specified
      square and direction
    -this method uses an associative matching algorithm to match the signal to closest
      labeled signal in the dictionary ddata.self.auditory_dict
    -a shallow ml algorithm, a deep algo or in fact a simply fuzzy logic algorithm is
      fine for this matching since it no longer forms the main basis for binding
      sensory information in the CCA3
    -binding is now done by matching not a label of a various sense, but rather the
      features of that sensory signal onto a navigation map
    -thus, this associative_match method provides a useful label for the signal that
      can be used for other things, but it is no longer the main means of binding
      sensory information into the representations within the CCA3

    -currently fuzzy associative matching library is used:
            from fuzzywuzzy import fuzz
            from fuzzywuzzy import process
            extract' method imported from fuzzywuzzy library

    input parameters:
    -sensory_input_one_direction which is the simulated auditory signal, ie,
      simulated_auditory_input_this_direction from simulated_auditory_to_vector() method
    -d ->
    -d.auditory_dict (called directly within the method)
        {'00000000':'strange silence', '11000000':'forest_noise', '11110000':'human cry help', '00010001':'smooth water sound', '11100000':'bird mating call', '01010101':'water spray noise'}

    output:
    -returns the most likely label for that auditory signal referencing d.auditory_dict and exact
      vector matched which is the key for that label
    '''
    # print('sensory_input_one_direction: ', sensory_input_one_direction)
    # print('d.auditory_dict: ', d.auditory_dict, '\n')
    # print('d.auditory_dict.keys(): ', d.auditory_dict.keys())
    top_matched_forest_keys_with_rating = process.extract(
        sensory_input_one_direction, d.auditory_dict.keys(), limit=1
    )
    # print('top_matched_forest_keys_with_rating: ', top_matched_forest_keys_with_rating)
    top_key = [top_matched_forest_keys_with_rating][0][0][0]
    label = d.auditory_dict[top_key]
    # print(label, top_key)
    return label, top_key


def generate_features_for_auditory_input(d, matched_sensory_vector: str):
    '''CCA3 ver
    -simulated_auditory_to_vector() produced a simulated auditory signal for specified
      square and direction
    -associative_match_auditory_input() uses an associative matching algorithm to match the signal
      to closest labeled signal in the dictionary ddata.self.auditory_dict
    -however, in the CCA3 we no longer bind a label but bind the real world features of a sensory
      input to a navigation map
    -since sensory inputs are being simulated (as opposed to using real video,audio, tactile, etc sensors)
      we need to generate, ie, simulate, the real world features for our auditory signal label
    -this binding information can then be used to bind the sensory input onto a navigation map
    -at present, we simply pull these features from a pre-constructed d.aud_features dictionary

    input parameters:
    -matched_sensory_vector is the matched audio input vector (ie, matched with dictionary above)
    -d ->
    -d.aud_features (called directly within the method)
        {'00000000':'xxxxxxxstrange silence', '11000000':'forest_noise', '11110000':'human cry help', '00010001':'smooth water sound', '11100000':'bird mating call', '01010101':'water spray noise'}

    output:
    -returns the features for this sensory signal and where on the 6x6 navigation map they belong
     [feature1, x1, y1, feature2, x2, y2, .....]
    '''
    print("matched_sensory_vector: ", matched_sensory_vector)
    print("d.aud_features: ", d.aud_features, "\n")
    features = d.aud_features[matched_sensory_vector]
    print(features)
    return features


def get_emulated_input(d, g, sense_inputs):
    '''gets input sensory vector depending on position of CCA1
    sense_inputs provides lists of possible sensory values for a particular position
    selector chooses which of these possible values to use, trying to emulate real world data input
    selector here is just expression 'random.randint(0, len(possibles[0])-1)' which equals
        [random index value for one of the inputs]  eg, if 3 possibles sensory inputs in the N
        position where CCA1 is, then index can be (0...2)
    parameters--
        d, g - project standard instantiations of ddata.py classes
        sense_inputs - eg, d.visual_database, d.auditory_database, etc
            the matrix with emulation of possible inputs of a given sense
            for a given square in the N,E,S, and W directions
    output--
        returns a list [N,E,S,W] containing 4 8-bit values, with each of
            these values corresponding to a sensory (for sense x) input
            for the CCA1's current position in the N, E, S and W direction
    TOTAL_ROWS = 6  #count EDGE squares
    TOTAL_COLS = 6  #count EDGE squares
    '''
    # check validity of d.cca1_position, do not allow to be in edge squares
    if d.cca1_position[0] <= 0 or d.cca1_position[1] <= 0:
        return [99999999, 99999999, 99999999, 99999999]
    if d.cca1_position[0] >= g.total_rows - 1 or d.cca1_position[1] - 1 >= g.total_cols:
        return [99999999, 99999999, 99999999, 99999999]

    # what are the adjacent square possible sensory input values in the N,E,S,W directions
    possibles = sense_inputs[d.cca1_position[0]][d.cca1_position[1]]

    # ensure that there is at least one sensory value to choose from in each of the
    # directions -- if not, then add g.filler (FILLER = '00000000') to that direction
    for i in range(3):
        if len(possibles[i]) == 0:
            print(
                "\ndebug: no input sensory value in direction ",
                i,
                " for current sensory matrix",
            )
            print("debug: thus following value will be used: ", g.filler)
            print("debug: current cca1 position is: ", d.cca1_position, "\n")
            possibles[i] = [g.filler]

    # now choose randomly one of the possible emulated sensory input values
    # do this in N, E, S, W directions -- gives [N, E, S, W]
    # == [0 index for N, 1 index for E, 2 index for S, 3 index for W]
    # thus, this is a list of emulated input sensory values from the N, E, S, W
    # directions of the CCA1's current position
    directional_input = [
        possibles[0][random.randint(0, len(possibles[0]) - 1)],
        possibles[1][random.randint(0, len(possibles[1]) - 1)],
        possibles[2][random.randint(0, len(possibles[2]) - 1)],
        possibles[3][random.randint(0, len(possibles[3]) - 1)],
    ]
    return directional_input


def get_hardware_input(d, g, sense_x_hardware_interface):
    '''gets input sensory vector in N, E, S and W directions surrounding the CCA1
    parameters--
        d, g - project standard instantiations of ddata.py classes
        sense_x_hardware_interface - specifies which hardware drivers to call
    output--
        returns a list [N,E,S,W] containing 4 8-bit values, with each of
            these values corresponding to a sensory (for sense x) input
            for the CCA1's current position in the N, E, S and W direction
    devpt note: pyboard physical cca1 device software is currently in palimpset.py
        and not actively being developed
                this method will current return an error vector with 99999999's
    '''
    # access hardware drivers
    print(g.version, g.hardware, d.cca1_position)
    if sense_x_hardware_interface not in ("pyboard0_interface", "pyboard1_interace"):
        print("\ndebug: warning: pyboard hardware implementation software")
        print("can be found in palimpset.py and is not supported here at present")
        print("directional_input, ie, sensory input,")
        print("is being set to 99999999 in all directions\n")
        input("please press any key to continue....\n")
        return [99999999, 99999999, 99999999, 99999999]
    return [99999999, 99999999, 99999999, 99999999]


def autonomic_reflex(d, g, h, input_vector: list, to_print: str) -> bool:
    '''
    autonomous modules as well as cca1 internal reflex centers
    processing an input for reflex activity

    parameters --
        d,g - project standard instantiation of ddata.py classes holding data structures
        to_print - message to print when method starts
        input_vector -  list of 4 vectors N,E,S,W
        [N, E, S, W]  N=8bit value eg, ['11011000', '11000010', '01111111', '11111100']
    output --
        True if a reflex escape motion occured
        False if no reflex was triggered
    '''
    # process input_vector
    print("\n-->", to_print)
    '''
    if not g.fastrun:
        if input("Enter 'y' to test reflex escape"):
            print('input_vector[0], danger to north, is being set to g.reflex_escape')
            input_vector[0] = g.reflex_escape
    '''

    # check to see if autonomic_reflex occurs
    for i in input_vector:
        if i == g.reflex_escape:
            # g.reflex_escape = REFLEX_ESCAPE = '10011001'
            print(
                "test of autonomic_reflex triggers with demonstration jump to the left if possible"
            )
            if d.cca1_position[1] == 1 or input_vector[3] == g.reflex_escape:
                print(
                    "cca1 is in left most square thus cannot jump to left, or else danger is from the west"
                )
                print("and thus do not want to jump left; other reflexes under devpt")
                return False
            print(
                "-->Escape motion to left triggered by north vector in input_vector: ",
                input_vector,
            )
            output_and_shaping(d, g, h, g.escape_left)
            # g.escape_left = ESCAPE_LEFT = '11111111'
            return True
    print("Input {} did not trigger any reflexes".format(input_vector))
    return False


def HLNs_sensory_process(
    sensory_input: list, references_dict, HLN_feedback=None, to_print=""
) -> list:
    '''subsymbolic identification of sensory inputs
    HLN_feedback affects identification choice
    parameters:
    -sensory_input is input_vectors_shaping [N, E, S, W] vector
    -references_dict is d.visual_dict if looking a visual inputs, for example
    -HLN_feedback is feedback circuits get for better identification as well as the
        transition to causality; in devp'take
    -to_print is message printed when the method runs
    output:
    -returns [N, E, S, W] vector where for each direction specify (eg):
         {'11111100': [100, 'obstruction'], '11100011': [62, 'lake']}
         -references_dict vectors 11111100 and 11100011 are the closest matches for
         sensory_input[N] (which actually was 11111100)
         -for the two closest vector matches also include a score of how close and
         the text that goes along with this vector
         thus vector is 4 dictionaries as such for each direction

    '''
    print("\n-->", to_print)
    print(
        "in HLNs_sensory_process--received sensory_input_visual/auditory/etc and HLN_feedback parameter: ",
        sensory_input,
        HLN_feedback,
    )
    print(
        "matching of N, E, S and W sensory_input vectors to descriptors in visual/auditory/etc reference dictionary"
    )
    north = associative_HLN_processing(sensory_input[0], references_dict)
    east = associative_HLN_processing(sensory_input[1], references_dict)
    south = associative_HLN_processing(sensory_input[2], references_dict)
    west = associative_HLN_processing(sensory_input[3], references_dict)
    return [north, east, south, west]


def associative_HLN_processing(
    sensory_input_one_direction: str, references_dict: dict
) -> dict:
    '''associative processing via HLNs (Hopfield-like Network units)
    at present call associative_processing which does this via
    simulation via more available artificial neural network frameworks
    -this method takes an input sensory vector, eg, 10011100, and then tries to match it against
    a dictionary of various input sensory vectors:label pairs, and then returns the two best
    matches, eg,  {'11111100': [75, 'obstruction'], '11000110': [75, 'forest']}
    parameters:
    -sensory_input_one_direction is a string representing the input sensory vector in a particular
        direction  eg,  10011100,
    -references_dict is d.visual_dict if looking a visual inputs, for example
    {'11100011': 'lake', '01010000': 'lost hiker visual', '11111100': 'obstruction',
    '00010001': 'shallow river', '00011001': 'shallow river + spraying water',
    '11000110': 'forest'}
     output:
    -returns {} dict for that direction, eg,
        {'11111100': [75, 'obstruction'], '11000110': [75, 'forest']}
    '''
    # other HLN functions, eg, setting up connections from maximal meaningfulness
    # can be implemented here as well
    MECH = "fuzzy"
    if MECH == "fuzzy":
        # print(associative_processing_via_fuzzy_mech(sensory_input_one_direction, references_dict))
        return associative_processing_via_fuzzy_mech(
            sensory_input_one_direction, references_dict
        )
    if MECH == "cnn":
        print("debug: warning: cnn not implemented in rewrite; check palimpset")
    if MECH == "hlnsiml":
        print(
            "debug: warning: simulation of HLNs pattern matching in palimpset; not implemented here"
        )
    return {
        "88888888": [99, "error non-existent nn"],
        "88888880": [99, "error -- non-existent nn"],
    }


def associative_processing_via_fuzzy_mech(
    sensory_input_one_direction: str, references_dict
) -> dict:
    '''fuzzy match in microsimulation to simulate NN in fuller simulation which in
    yet fuller simulation (ie, more fine grained model) simulates network of HLNs
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    -this method takes an input sensory vector, eg, 10011100, and then tries to match it against
    a dictionary of various input sensory vectors:label pairs, and then returns the two best
    matches, eg,  {'11111100': [75, 'obstruction'], '11000110': [75, 'forest']}
    parameters:
    -sensory_input_one_direction is a string representing the input sensory vector in a particular
        direction  eg,  10011100,
    -references_dict is d.visual_dict if looking a visual inputs, for example
    {'11100011': 'lake', '01010000': 'lost hiker visual', '11111100': 'obstruction',
    '00010001': 'shallow river', '00011001': 'shallow river + spraying water',
    '11000110': 'forest'}
     output:
    -returns {} dict for that direction, eg,
        {'11111100': [75, 'obstruction'], '11000110': [75, 'forest']}
        {'sensory 8 bit' : [likelihod, description], 'sensory 8 bit': : [likelihod, description]}
    -'extract' method imported from fuzzywuzzy library
    '''
    # print('sensory_input_one_direction: ', sensory_input_one_direction)
    # print('references_dict: ', references_dict, '\n')
    # print('references_dict.keys(): ', references_dict.keys())
    top_matched_forest_keys_with_rating = process.extract(
        sensory_input_one_direction, references_dict.keys(), limit=2
    )
    print("top_matched_forest_keys_with_rating: ", top_matched_forest_keys_with_rating)
    # sensory_input_one_direction = '11111101'
    # references_dict =  {'11100011': 'lake', '01010000': 'lost hiker visual', '11111100': 'obstruction',
    #   '00010001': 'shallow river', '00011001': 'shallow river + spraying water', '11000110': 'forest'}
    # top_matched_forest_keys_with_rating = [('11111100', 88), ('11100011', 62)]
    alpha_matches = {}
    for topkey in top_matched_forest_keys_with_rating:
        # topkey =  ('11111100', 88)
        # topkey[0] = '11111100'  therefore alpha_matches = {'11111100': [topkey[1], references_dict[topkey[0]]])
        #                        therefore alpha_matches = {'11111100': [88, 'obtruction']}
        alpha_matches[topkey[0]] = [topkey[1], references_dict[topkey[0]]]
        # after first loop:  alpha_matches = {'11111100': [88, 'obstruction']}
        # after 2nd loop:      = {'11111100': [88, 'obstruction'], '11100011': [62, 'lake']}
    print("associative top matches: ", alpha_matches)
    return alpha_matches


def sensory_fuse(to_print, sensory1, sensory2, HLN_feedback=None):  # DEPRECATED
    '''DEPRECATED
    fusion of processed sensory inputs
    simple Nano-level fusion -- adds values of top choices of two senses
    however this type of binding often causes false positives
    DEPRECATED
    '''
    if HLN_feedback:
        pass
    fused_senses = [{}, {}, {}, {}]
    print("\n-->", to_print)
    for j in range(4):
        fused_senses[j] = dict(sensory1[j])
        for i in sensory2[j]:
            if i in fused_senses[j]:
                fused_senses[j][i][0] = fused_senses[j][i][0] + sensory2[j][i][0]
            else:
                fused_senses[j][i] = sensory2[j][i][0:2]
    return fused_senses


def fused_select(fused_vector, HLN_feedback=None):  # DEPRECATED
    '''DEPRECATED
    uses meaningfulness as well as hierarchical feedback to determine which of
    the sensory-binding sensory possibilities will be selected
    nb see note/print statements about Nano model
    USED AFTER SENSORY_FUSE -->  DEPRECATED
    '''
    fused_vector = fused_vector[0]
    if HLN_feedback:
        print("In current Nano model of CCA1 the max value of all the possible")
        print(" sensory bindings is used. This is updated with CCA1-related")
        print(
            " mechanisms (meaningfulness as well as hierarchical feedback) in Micro model."
        )
    max_record = -1
    current_max = -1  # use for readability
    for i in fused_vector:
        if fused_vector[i][0] >= current_max:
            current_max = fused_vector[i][0]
            max_record = i
    return max_record, fused_vector[max_record]


def NESW_fused_select(fused_vector):  # DEPRECATED
    '''DEPRECATED
    calls fused_select for each of the NESW components
    returns eg--  01010101 [100, 'lost hiker']
    USED AFTER FUSED_SELECT --> DEPRECATED
    '''
    NESW_max_record = [0, 0, 0, 0]
    NESW_fused_vector = [0, 0, 0, 0]
    for i in range(4):
        a, b = fused_select(fused_vector)
        NESW_max_record[i] = a
        NESW_fused_vector[i] = b
    return NESW_max_record, NESW_fused_vector


def apply_meaningfulness(h, a_precombos):
    '''
    applies meaningfulness to sensory input
    currently if meaningfulness True then accepts
     current methods, if False degrades slightly
    todo: port and drop in fully functioning
     meaningfulness component
    '''
    # currently for nano deprecation scaffolding work assume
    # current sensory processing is applying meaningfulness
    if h.meaningfulness:
        print("directional input sensory vector: ", a_precombos)
        return a_precombos

    # currently for nano deprecation scaffolding work slightly
    # degrade sensory_input since no meaningfulness
    print("directional input sensory vector: ", a_precombos, "-> ")
    a_precombos = a_precombos[:1] + "0" + a_precombos[2:]
    print(a_precombos)
    return a_precombos


def HLNs_sensory_fusion(d, h, *senses_to_fuse, HLN_feedback=None, to_print=""):
    # CURRENTLY BEING USED
    # d.fused_matches = HLNs_sensory_fusion(d, sensory_input_visual, sensory_input_auditory, d.fused_dict, None, 'SENSORY FUSION')
    '''fusion of processed sensory inputs
    nb replaces original sensory_fuse
    simple Nano-level fusion
    senses_to_fuse - the difference senses this method should fuse
    eg, sensory1 is sensory_input_visual
    eg, sensory2 is sensory_input_auditory
    combine 8 bits of sensory1 (eg, visual) with 8 bits of sensory2
      (eg, auditory) to yield 16 bit vector: visual(8)+auditory(8)
    combine the various possible combinations sensory1 and sensory2 components
    then compare the 16 bit combos against {fused_dict} to see most
      likely matches
    can generate combos from matching outputs of each sense
    however in this version just concatenate the ref sensory output

    eg, example of d.fused_matches = HLNs_sensory_fusion output:
    [
    {   '1111110000000000': [94, 'obstruction + strange silence -> EDGE'],
        '1110001100010001': [75, 'lake + smooth water sound -> lake']           },
    {   '1100011011000000': [88, 'forest + forest noise -> forest'],
        '0101000011110000': [88, 'lost hiker visual + human cry help -> hiker'] },
    {   '1111110000000000': [81, 'obstruction + strange silence -> EDGE'],
        '1100011011000000': [75, 'forest + forest noise -> forest']             },
    {   '1111110000000000': [94, 'obstruction + strange silence -> EDGE'],
        '1110001100010001': [75, 'lake + smooth water sound -> lake']           }
    ]
    usage:
    d.fused_matches = HLNs_sensory_fusion(d, sensory_input_visual, sensory_input_auditory, d.fused_dict, None, 'SENSORY FUSION')
    '''
    # set up senses_to_fuse, start of method
    # at this time method simply used to fuse visual and auditory inputs
    # possible inputs include
    # nb use last word in name as the parameter, eg, 'd.sensory_input_visual' thus specify 'visual'
    # d.sensory_input_visual
    # d.sensory_input_auditory
    # d.sensory_input_tactile
    # d.sensory_input_olfactory
    # d.sensory_input_chemical
    # d.sensory_input_pressure
    # d.sensory_input_balance
    # d.sensory_input_damage
    # d.sensory_input_orientation
    # d.sensory_input_accelerometer
    # d.sensory_input_gps
    # d.sensory_input_radar
    # d.sensory_input_lidar
    # d.sensory_input_rf
    # d.sensory_input_electroreception
    # d.sensory_input_magnetoreception
    print(senses_to_fuse)
    if len(senses_to_fuse) < 2:
        print(
            "must specify more senses to fuse, thus senses_to_fuse set to ('visual', 'auditory')"
        )
        senses_to_fuse = ("visual", "auditory")
    if len(senses_to_fuse) >= 3:
        print(
            "at this time this method fuses only two senses of visual and auditory, thus senses_to_fuse set to ('visual', 'auditory')"
        )
        senses_to_fuse = ("visual", "auditory")
    if "visual" in senses_to_fuse and "auditory" in senses_to_fuse:
        print("fusion of specified visual and auditory inputs will now occur")
    else:
        print(
            "at this time this method fuses visual and auditory, thus senses_to_fuse set to ('visual', 'auditory')"
        )
        senses_to_fuse = (
            "visual",
            "auditory",
        )  # set for future multi-sensory capabilities of the this method
    sensory1 = d.sensory_input_visual
    sensory2 = d.sensory_input_auditory
    ref_values = (
        d.fused_dict
    )  # current d.fused_dict is visual and auditory fused; will deprecate later for better map based representation in cca2
    print("\n-->", to_print)

    # handle feedback to the hierarchy of hopfield-like network units
    # at this time placeholder code
    if HLN_feedback:
        pass

    # create combos of 16 bit words
    # this implementation just concatenates the raw auditory and visual inputs, thus,
    # we are not really taking advantage of the lower level processing
    precombos = [[], [], [], []]
    combos = [[], [], [], []]
    for i in range(4):
        precombos[i] = sensory1[i] + sensory2[i]
        combos[i] = apply_meaningfulness(h, precombos[i])
        if HLN_feedback:
            print(
                "Todo in current rewrite: apply meaningfulness as well as hierarchical feedback"
            )

    # now must look for best matches in concatenated ref inputs dictionary
    d.fused_matches = [[], [], [], []]
    for i in range(4):
        top_matched_forest_keys_with_rating = process.extract(
            combos[i], ref_values.keys(), limit=2
        )
        # eg,  [('1111110000000000', 94), ('1100011011000000', 75)]
        alpha_matches = {}
        for topkey in top_matched_forest_keys_with_rating:
            # eg, first iteration: ('1111110000000000', 94)
            # eg, second iteration:  ('1100011011000000', 75)
            alpha_matches[topkey[0]] = [
                topkey[1],
                ref_values[topkey[0]],
            ]  # create dictionary {key:[rating, descriptor]}
            # eg, first iter'n: {'1111110000000000': [94, 'obstruction + strange silence -> edge']}
            # eg, second iter'n: {'1111110000000000': [94, 'obstruction + strange silence -> edge'], '1100011011000000': [75, 'forest + forest noise -> forest']}
        d.fused_matches[i] = alpha_matches
        # [{N1, N2}, {E1, E2}, {S1, S2}, {W1, W2}]  N1= 16bit:[rating, descriptor]
        # i==0:  [{'1111110000000000': [94, 'obstruction + strange silence -> edge'], '1100011011000000': [75, 'forest + forest noise -> forest']}, [], [], []]
        # i==1:  [{'1111110000000000': [94, 'obstruction + strange silence -> edge'], '1100011011000000': [75, 'forest + forest noise -> forest']}, {'1100011011000000': [88, 'forest + forest noise -> forest'], '0101000011110000': [81, 'lost hiker visual + human cry help -> hiker']}, [], []]

    # for devpt print out variables
    flag_a = True
    if flag_a:
        print("------------------------------------------------")
        print("sensory_input_visual: ", sensory1)
        print("sensory_input_auditory: ", sensory2)
        print("16bit concatenated values association dictionary:\n", ref_values, "\n")
        fused = d.fused_matches
        print("\n>>fused sensory signal: ", fused, "\n")
        # eg, [{'1111110000000000': [94, 'obstruction + strange silence -> EDGE'],
        #      '1100011011000000': [..]},
        #       {..}, {..}, {..}]
        print("N : ", fused[0])
        print("E : ", fused[1])
        print("S : ", fused[2])
        print("W : ", fused[3], "\n")

    # eg, print(fused[0]['1111110000000000'][1]) -- gives North dictionary, key:value with this key, descriptor in value portion
    return d.fused_matches


def NESW_fused_select2(d, g) -> tuple:
    # CURRENTLY BEING USED
    # usage: d.max_fused_index, d.max_fused_value = NESW_fused_select2(g, d)
    '''in each direction, d.fused_matches may have 2 (or more) directionary items as follows:
    {'16bit sensory input':[match score, 'string description', ....}
    -this method returns, for each direction, the item 16bit:[score, description] which has
    the highest score in each direction
    -see comments below to follow through an example, although only N direction shown
    parameters:
        g - project standard instantiation of class holding data structures
        d with d.fused_matches - value returned from HLNs_sensory_fusion method
    returns:
        NEWS_max_record -> d.max_fused_index = 16bit sensory input string in each direction that corresponds to maximal associative match numerical score
        NESW_fused_vector -> d.max_fused_value = for each direction [match numeric score, description string
    '''
    NESW_max_record = [0, 0, 0, 0]
    NESW_fused_vector = [0, 0, 0, 0]
    # we will follow the processing in N direction of d.fused_matches input through this method below
    # eg, d.fused_matches is: [
    # {'1111110000000000': [94, 'obstruction + strange silence -> EDGE'], '1100011011000000': [75, 'forest + forest noise -> forest']},
    # {'1100011011000000': [81, 'forest + forest noise -> forest'], '0101000011110000': [81, 'lost hiker visual + human cry help -> hiker']},
    # {'1110001100010001': [88, 'lake + smooth water sound -> lake'], '1111110000000000': [75, 'obstruction + strange silence -> EDGE']},
    # {'1111110000000000': [88, 'obstruction + strange silence -> EDGE'], '1110001100010001': [69, 'lake + smooth water sound -> lake']}    ]
    for direction in range(4):
        # direction is:  0
        # direction is:  1 .... and repeats again for this east direction, and the south and west directions
        fused_vector_unidirection = d.fused_matches[direction]
        # d.fused_matches[direction] is:  {'1111110000000000': [94, 'obstruction + strange silence -> EDGE'],
        #                               '1100011011000000': [75, 'forest + forest noise -> forest']}
        max_record = -1
        current_max = -1  # use for readability
        for match_16bit in fused_vector_unidirection:
            # match_16bit is:  1111110000000000 for 1st pass for N direction
            # match_16bit is:  1100011011000000 for 2nd pass for N direction
            # fused_vector_unidirection[match_16bit][0] is:  94 and current_max is:  -1  #1st pass for N direction
            # fused_vector_unidirection[match_16bit][0] is:  75 and current_max is:  94  #2nd pass for N direction
            if fused_vector_unidirection[match_16bit][0] >= current_max:
                current_max = fused_vector_unidirection[match_16bit][0]
                max_record = match_16bit
                # print('current_max is: ', current_max, ' and max_record is: ', max_record)
                # current_max is:  94  and max_record is:  1111110000000000
                # this logic is not executed for 2nd pass for N direction since 75 is not >= 94 the value of current_max

        NESW_max_record[direction] = max_record
        # NESW_max_record[direction] = max_record is:  1111110000000000
        # thus this is the value that goes into the N component of NESW_max_record = [0, 0, 0, 0] initial values
        NESW_fused_vector[direction] = fused_vector_unidirection[max_record]
        # NESW_fused_vector[direction] = fused_vector_unidirection[max_record] is : [94, 'obstruction + strange silence -> EDGE']
        # thus this is the value that goes into N component of NESW_fused_vector = [0, 0, 0, 0] initial value

    if g.sensory_buffer != []:
        print("in NEWS_fused_select and feeding back sensory_buffer to sensory inpu")
        print("\nsensory_buffer has a value from the causal memory which will be")
        print(" used as sensory input instead of subsymbolic values used otherwise:")
        print(
            " previous NESW_max_record, NESW_fused_vector: ",
            NESW_max_record,
            "\n",
            NESW_fused_vector,
        )
        NESW_max_record = g.sensory_buffer[0]
        NESW_fused_vector = g.sensory_buffer[1]
        print(
            " new NESW_max_record, NESW_fused_vector: ",
            NESW_max_record,
            "\n",
            NESW_fused_vector,
        )
        g.sensory_buffer = []

    # NESW_max_record = [N, E, W, S]  eg, N = '1111110000000000'
    # NESW_fused_vector = [N, E, W, S] eg, N = [94, 'obstruction + strange silence -> EDGE']
    # usage: d.max_fused_index, d.max_fused_value = NESW_fused_select2(g, d)
    # thus, d.max_fused_index  = ['1111110000000000', '0101000011110000', '1110001100010001', '1111110000000000']
    # thus, d.max_fused_value  = [
    # [94, 'obstruction + strange silence -> EDGE'],
    # [81, 'lost hiker visual + human cry help -> hiker'],
    # [88, 'lake + smooth water sound -> lake'],
    # [88, 'obstruction + strange silence -> EDGE']    ]
    # -NEWS_max_record -> d.max_fused_index = 16bit sensory input string in each direction that corresponds to maximal associative match numerical score
    # -NESW_fused_vector -> d.max_fused_value = for each direction [match numeric score, description string
    return NESW_max_record, NESW_fused_vector


def get_current_autonomic(d, g, to_print, influence=None):
    '''returns current_autonomic
    Nano-level
    instincts are influenced by external sensory plus internal autonomic sensory
    'influence' allows caller to influence what autonomic result should be returned
    Micro-level and above really should interface with wake-sleep method
    each call increments age of system 'age_autonomic_calls'
    current_autonomic is global since can affect much of system
    '''
    print("\n-->", to_print)
    # autonomic, age, instinctual states can affect many parts of the system, and kept global
    d.age_autonomic_calls += 1
    #
    # caller can specify autonomic by simply entering it as influence parameter
    if influence:
        aa = str(influence)
        aa = aa + g.filler
        aa = aa[0:8]  # Nano level
        print("current_autonomic is value is set by caller of method: ", aa)
        return aa
    #
    # duty cycle specifies how often to change current_autonomic in the simple Nano model
    DUTY_CYCLE_CHANGE_AUTONOMIC = 5
    duty_cycle_random = random.randint(1, DUTY_CYCLE_CHANGE_AUTONOMIC)
    #
    # randomly change, if duty cycle allows, current_autonomic in simple Nano model
    if d.age_autonomic_calls <= 1 or duty_cycle_random == 1:
        d.current_autonomic = random.choice(list(d.autonomic_dict.keys()))
        print(
            "current_autonomic value randomly changed to: ",
            d.current_autonomic,
            " which corresponds to autonomic state of ",
            d.autonomic_dict[d.current_autonomic],
        )
        return d.current_autonomic
    # otherwise current_autonomic will not be changed
    print(
        "current_autonomic remains as {} ".format(d.autonomic_dict[d.current_autonomic])
    )
    return d.current_autonomic


def get_current_instinct(d, g, h, to_print):
    '''each evaluation cycle the processed sensory input vector is propagated to the
    instinctual core goals module and to the causal group of HLNs
    the processed sensory input vector may trigger in the instinctual core goals module
     intuitive logic/physics/psychology/goal planning procedural vectors which are
     propagated to the motor output group of HLNs as well as to the causal group of HLNs
    -in LAMPREY version at simplest present level get back goal of 'forward/eat/goal' or
     'conserve energy' if energy stores very low -- this goes to hippocampus module and
     effectively allows one of the skewed walk or simple planning vectors then to go to the
     motor output
    -in HUMAN version we need more causal behavior from the instinctual core goals module,
     including the activation of learned logic/physics/psychology/goal planning procedural
     vectors
    -this method directs to the appropriate get_current_instinct method
    usage:
    d.current_instinct = get_current_instinct(d, g, d.max_fused_index, 'GET CURRENT INSTINCT')
    eg, d.max_fused_index = [N, E, W, S]  eg, N = '1111110000000000'
    eg, to_print -- message to print out
    '''
    if h.current_hippocampus in ["LAMPEY", "FISH", "REPTILE", "MAMMAL"]:
        return get_current_instinct1(d, to_print)
    if h.current_hippocampus in ["HUMAN", "SUPERINTELLIGENCE", "SUPERINTELLIGENCE2"]:
        return get_current_instinct5(d, g, h, to_print)
    print("coding issue in get_current_instinct")
    return get_current_instinct1(d, to_print)


def get_current_instinct1(d, to_print):
    '''returns current_instinct from the instinctual core
    goals module
    Nano-level
    instincts are influenced by external sensory plus internal autonomic sensory
    #to help see what instinct and other values are keyed as
    instinct = {'10000000':'forward eat/goal', '11000000':'left avoid', '01000000':'right avoid', '00000000':'conserve energy' }
    autonomic = {'00000000':'conserve energy', '00000001':'move to different area',
                 '00000010':'eat', '00000011':'reproduce'}
    '''
    print("\n-->", to_print + "**inside get_curr_inst1**")
    print(
        "nb. instinct value calculated here and can be used in subsequent methods as desired"
    )
    # at Nano level consider very basic instinctual strategies
    # | current_autonomic + d.max_fused_index -> current_instinct |
    # these strategies can at Micro level be replaced by neural network or fuzzy implementation
    #
    # autonomic, age, instinctual states can affect many parts of the system, and kept global
    #
    # if human cry sensory input then search robot should move regardless of energy/other issues
    if d.max_fused_index == "11110001":
        print(
            "'11110001' human cry help --thus return instinct '10000000' -- 'forward eat/goal'"
        )
        return "10000000"
    #
    # if autonomic is conserve energy and no human cry then instinct to conserve energy also
    if d.current_autonomic == "00000000":
        print(
            "current_autonomic is {} thus return instinct value: {}".format(
                d.current_autonomic, d.instinct_dict["00000000"]
            )
        )
        # conserve energy
        return "00000000"
    #
    # for now, for other autonomic values, treat as CCA1 is active and in process of
    # wanting to reach goal, thus, we will return an instinct value of 1000 0000 which
    # is 'forward eat/goal'
    print(
        "instinct method note: despite other auton values will treat as CCA1 is active"
    )
    return "10000000"


def get_current_instinct5(d, g, h, to_print, noncausal_instinct=True, directional=0):
    '''returns current_instinct from the instinctual core goals module
    Nano-level-- simpler pattern matching used; PyTorch can be dropped into micro version
    -if noncausal_instinct is True then just calls lower level get_current_instinct method
      to return an instinct based on autonomic and other non-causal factors
    -however, if noncausal_instinct is False, then this function will help the causal
     aspects of the hippocampus to make causal decisions
    #
    #instinct values are goal action values to shape output
    instinct = {'10000000':'forward eat/goal', '11000000':'left avoid', '01000000':'right avoid', '00000000':'conserve energy'}
    #autonomic values will shape instinct value along with sensory input values
    autonomic = {'00000000':'conserve energy', '00000001':'move to different area',
                 '00000010':'eat', '00000011':'reproduce'}
    #intuit_instinct values are procedural vectors triggered in intuitive logic/physics/psychology or goal plannig
    intuitive_instinct_dict = {'1111110000000000':'EDGE_logic',
                          '1110001100010001':'water_everywhere_logic',
                          '0001100101010101':'water_everywhere_logic'}
    #d.learned_instinct_dict values are procedural vectors triggered in learned logic/physics/psychology or goal plannig
    d.learned_instinct_dict = {<eg>}
    #
    '''
    # d.current_instinct    #future use -- will affect both intuitive and learned triggering
    # dintuitive_instinct_dict  #future use -- modify if learning <-  to deprecated and port over still
    # d.learned_instinct_dict    #future use -- modify if learning <-  to deprecated and port over still

    # if noncausal_instinct==True then this is call from evaluation_cycles to get instinct based
    #    on auton and other non-causal factors
    h.pass_h()
    if noncausal_instinct:
        print(
            "\n-->",
            to_print + "**inside get_curr_inst5 noncausal_instinct==True call**",
        )
        print(
            "since this is a noncausal instinct call, will in turn call get_current_instinct1()"
        )
        return_value = get_current_instinct1(d, "")
        print(
            "just returned back to curr_inst5 from current_instinct1 with a value of: ",
            return_value,
        )

    # if noncausal_instinct==False then returns procedural vector from intuitive & learned instinct
    else:
        # no reason to avoid move in this direction if nothing below is triggered
        print("\n-->", to_print + "**inside get_curr_inst5**")
        return_value = True

        # trigger instinctual core goals module and see what procedural vectors are returned
        if return_value is not False:
            return_value = instinct_triggering(
                d,
                g,
                d.max_fused_index[directional],
                sensory_input_vector2="",
                learnmode=False,
                directional=directional,
            )
            print("instinct5 will now return: ", return_value)

        # trigger learned pre/causal memory and see what procedural vectors are returned
        if return_value is not False:
            return_value = learned_triggering(
                d,
                g,
                d.max_fused_index[directional],
                sensory_input_vector2="",
                learnmode=False,
                directional=directional,
            )
            print("instinct5 will now return: ", return_value)

    return return_value


def instinct_triggering(
    d,
    g,
    sensory_input_vector1,
    sensory_input_vector2="",
    learnmode=False,
    directional="00",
):
    '''returns procedural vector from instinctive core goals module in response to one or more
    fused and/or coomponent sensory vectors
    -nano - simpler pattern matching used; PyTorch can be dropped into micro version
    -if learnmode==True then some learning is possible with regard to the triggering
     intuitive logic/physics/psychology/planning, but not as extensive as with learned components
    -the autonomic age will affect the developmental timer and activation of various portion
    of which vectors become triggered
    #
    #instinct values are goal action values to shape output
    instinct = {'10000000':'forward eat/goal', '11000000':'left avoid', '01000000':'right avoid', '00000000':'conserve energy'}
    #autonomic values will shape instinct value along with sensory input values
    autonomic = {'00000000':'conserve energy', '00000001':'move to different area',
                 '00000010':'eat', '00000011':'reproduce'}
    #intuit_instinct values are procedural vectors triggered in intuitive logic/physics/psychology or goal plannig
    intuitive_instinct_dict = {'1111110000000000':'EDGE_logic',
                          '1110001100010001':'water_everywhere_logic',
                          '0001100101010101':'water_everywhere_logic'}
    #d.learned_instinct_dict values are procedural vectors triggered in learned logic/physics/psychology or goal plannig
    d.learned_instinct_dict = {<eg>}
    #
    '''
    # d.current_instinct    #future use -- will affect both intuitive and learned triggering
    # d.intuitive_instinct_dict  #future use -- modify if learning <-  to deprecated and port over still
    # d.learned_instinct_dict    #future use -- modify if learning <-  to deprecated and port over still

    return_value = True
    if learnmode:
        print("learnmode true, also sens2: ", sensory_input_vector2)

    top_match = process.extract(
        sensory_input_vector1, d.intuitive_instinct_dict.keys(), limit=1
    )
    # print('top_match[0][0]: ', top_match[0][0], 'top_match[0][1]: ', top_match[0][1])
    # checkpoint_beep(3, 'at topmatch')
    print(
        "sensory input is: {}, top_match is: {}".format(
            sensory_input_vector1, top_match
        )
    )
    if top_match[0][1] < 85:
        print(
            "top_match is less than 85% thus no modification of instinct_triggering return_value\n"
        )
    else:
        # print('top_match is more than 85% thus trigger intuitive code')

        if d.intuitive_instinct_dict[top_match[0][0]] == "EDGE_logic":
            # checkpoint_beep(3)
            g.gconscious(
                ["....but this is an EDGE thus hippo_calc will be False", directional]
            )
            print("....but this is an EDGE thus hippo_calc will be False")
            return_value = False

        if d.intuitive_instinct_dict[top_match[0][0]] == "water_everywhere_logic":
            # checkpoint_beep(3, 'checkpoint -- water damage may occur -- click ENTER to continue')
            print(
                "....but this is water danger thus try again navigation for this move"
            )
            g.gconscious(
                [
                    "....but this is a lake thus try again navigation for this move",
                    directional,
                ]
            )
            return_value = False

    # print('leaving instinct_triggering(), sensory_input_vector is {}, return is {}'.format(sensory_input_vector1, return_value))
    return return_value


def learned_triggering(
    d,
    g,
    sensory_input_vector1,
    sensory_input_vector2="",
    learnmode=False,
    directional="00",
):
    '''returns procedural vector from instinctive core goals module in response to one or more
    fused and/or coomponent sensory vectors
    -nano - simpler pattern matching used; PyTorch can be dropped into micro version
    -if learnmode==True then learning is possible with regard to the triggering
     learning and availability of logic/physics/psychology/planning procedural vectors
    -the autonomic age will affect the developmental timer and activation of various portion
    of which vectors become triggered, but not as systematically as with regard
    to the intuitive ones
    #
    #instinct values are goal action values to shape output
    instinct = {'10000000':'forward eat/goal', '11000000':'left avoid', '01000000':'right avoid', '00000000':'conserve energy'}
    #autonomic values will shape instinct value along with sensory input values
    autonomic = {'00000000':'conserve energy', '00000001':'move to different area',
                 '00000010':'eat', '00000011':'reproduce'}
    #intuit_instinct values are procedural vectors triggered in intuitive logic/physics/psychology or goal plannig
    intuitive_instinct_dict = {'1111110000000000':'EDGE_logic',
                          '1110001100010001':'water_everywhere_logic',
                          '0001100101010101':'water_everywhere_logic'}
    #d.learned_instinct_dict values are procedural vectors triggered in learned logic/physics/psychology or goal plannig
    d.learned_instinct_dict = {'0001000100010001':'test_case_learned'}
    #
    '''
    # d. current_instinct    #future use -- will affect both intuitive and learned triggering
    # d.intuitive_instinct_dict  #future use -- modify if learning <-  to deprecated and port over still
    # d.learned_instinct_dict    #future use -- modify if learning <-  to deprecated and port over still

    # g.sensory_buffer       #pre/causal memory can feed partial results back as a sensory input
    return_value = True
    if learnmode:
        print(
            "values sent to learned_triggering: ",
            sensory_input_vector1,
            sensory_input_vector2,
            directional,
            learnmode,
        )

    if g.sensory_buffer != []:
        print(
            "coding error -- non-empty sensory_buffer at start of learned_triggering()",
            g.sensory_buffer,
        )
        g.sensory_buffer = []
        print("sensory_buffer reset\n")

    print(
        "inside learned_triggering now -- set up for demo triggering and causal feedback of intermediate result"
    )
    print(" to input sensory stages; learning todo")
    top_match = process.extract(
        sensory_input_vector1, d.learned_instinct_dict.keys(), limit=1
    )
    print(
        "top_match[0][0] of d.learned_instinct_dict: ",
        top_match[0][0],
        "top_match[0][1]: ",
        top_match[0][1],
    )
    print(
        "sensory input is: {}, top_match is: {}".format(
            sensory_input_vector1, top_match
        )
    )
    if top_match[0][1] < 85:
        print(
            "top_match of d.learned_instinct_dict is less than 85% thus no modification of return_value"
        )
    else:
        print("top_match is more than 85% thus learned code")

        if d.learned_instinct_dict[top_match[0][0]] == "test_case_learned":
            print("just triggered test_case_learned")
            g.gconscious(["....just triggered test_case_learned", directional])
            g.sensory_buffer = [
                [
                    "1111110000000000",
                    "1100011011000000",
                    "1100011011000000",
                    "1111110000000000",
                ],
                [[88, "EDGE"], [88, "forest"], [88, "forest"], [88, "EDGE"]],
            ]
            print(
                "\n....inside learned_triggering()....will return True and continue with"
            )
            input("DEBUG DEBUG AT LEARNED TRIGGERING CRASH AREA")
            print(
                " whatever move would have occurred, but next move the intermediate result"
            )
            print(
                " being fed back to the input_sensory stages (simulated EDGE-forest-EDGE-EDGE)"
            )
            print(
                " will be processed instead of whatever sensory is occuring at that time."
            )
            print(
                " (please note the flexibility of given a causal feedback to the sensory input"
            )
            print(
                " stages, in other cases it would be more appropriate for the move this evaln"
            )
            print(
                " to be paused; similarly in other cases the fed back causal signal does not"
            )
            print(
                " have to take precedence over the fed back causal signal, but could override"
            )
            print(" it.\n")
            return_value = True
            # see explanation above why True is being returned

    print(
        "leaving learned_triggering(), sensory_input_vector is {}, return is {}".format(
            sensory_input_vector1, return_value
        )
    )
    print("value of sensory_buffer is: ", g.sensory_buffer, "\n")
    return return_value


def pattern_memory(d, verbose=0):
    '''additional memory area
    generally holds non-procedural information
    nano version -- scaffolding to drop in more authentic component
    '''
    if verbose:
        print("CHECKPOINT: in pattern memory method", d.max_fused_index)
    return True


def emotional(d, verbose=0):
    '''influences moves and goals
    influences learning of new facts and procedures
    allows effective learning of infrequent events and obviates the
     class imbalance problem seen in conventional neural networks
    --> rare events can sometimes be very important events to learn
    nano version -- scaffolding to drop in more authentic component
    '''
    if verbose:
        print("CHECKPOINT: in emotional method", d.max_fused_index)
    return True


def seq_and_error(d, verbose=0):
    '''sequential/error-correcting memory is an optional moduleand
    distinct from memories in MANNs or other NN
    useful for detecting changes in a spectrum of external and internal data,
    and storing learning sequences which can automatically be repeated later as needed
    nano version -- scaffolding to drop in more authentic component
    '''
    if verbose:
        print("CHECKPOINT: in sequential and error method", d.max_fused_index)
    return True


def update_hippoc_int_map(d):
    '''spatial maps and positioning for CCA1
    builds internal map associated with hippocampus2
    '''
    # build internal int_map_update
    for i, j in ((0, "00"), (1, "01"), (2, "10"), (3, "11")):
        if d.max_fused_index[i] == "1100011011000000":  # forest
            int_map_update(d, j, "forest")
        if d.max_fused_index[i] == "0001000100010001":  # sh_rvr
            int_map_update(d, j, "sh_rvr")
        if d.max_fused_index[i] == "0001100101010101":  # wtrfall
            int_map_update(d, j, "wtrfall")
        if d.max_fused_index[i] == "0101000011110000":  # lost hiker
            int_map_update(d, j, "hiker")
        if d.max_fused_index[i] == "1111110000000000":  # EDGE
            int_map_update(d, j, "EDGE")
        if d.max_fused_index[i] == "1110001100010001":  # lake
            int_map_update(d, j, "lake")
    d.print_int_map_previous()
    d.print_int_map()
    d.print_forest_map()
    return True


def hippocampus2(d, g, h):
    '''spatial maps and positioning for CCA1
    #
    hippocampus 1. Lamprey hippocampal/brain analogue - note: will default to quasi-skewed walk')
    hippocampus 2. Fish hippocampal/telencephalon analogue - note: currently reverts back to lamprey')
    hippocampus 3. Reptile hippocampal/pallium analgoue - note: some precausal features')
    hippocampus 4. Mammalian hippocampus - note: currently reverts back to reptile')
    hippocampus 5. Human hippocampus - note: pending status, some simple causal features')
    hippocampus 6. Superintelligence level 1 - note: currently reverts back to lower human level')
    hippocampus 7. Superintelligence level 2 - note: currently reverts back to lower human level')
    #
    --quick review of what hippocampus in biology does (since otherwise literature confusing for
    non-bio background reader due to terminology):
    -mammals have left and right hippocampi
    -in mammals part of the 'allocortex'=='heterogenetic cortex' versus the 'neocortex'
    (neocortex 6 layers vs. 3-4 cell layers in allocortex; types of allocortex: paleocortex,
    archicortex and transitional (ie, to neocortex) periallocortex)
    olfactory system also part of the allocortex
    -considered part of 'limbic system' (=='paleomammalian cortex' midline structures
     on L and R of thalamus, below temporal lobe; involved in emotion, motivation,
     olfactory sense and memory; making memories affected by limibc system)
     (basic limbic system is really amygdala, mammillary bodies, stria medull, nuc Gudden,
     but also tightly connected to limbic thalamus, cingulate gyrus, hippocampus,
     nucleus accumbens, anterior hypothalamus, ventral tegmental area, raphe nuc,
     hebenular commissure, entorhinal cortex, olfactory bulbs)
    -hippocampus involved in spatial memory, in this CCA1 simulation that is what similarly
    name method does
    -hippcampus also involved in putting together portions of memory throughout whole brain
    -given relation between learning and memory, not surprising to find that hippocampus involved
    in learning; in more advanced hippocampal methods, ie, beyond LAMPREY level
    these indeed are implemented in this method
    -hippocampus needed for consolidation of (certain types) of short-term memories
    into long-term memory, and for spatial navigation
    -above note that in primates hippocampus is in bottom part of medial temporal lobe
    -hippocampus = 'hippocampus proper'==Ammon's horn + dentate gyrus
    -hippocampus in all mammals; animals with better spatial memories are found to have
    larger hippocampal structures
    -in other vertebrates (ie, fish to reptiles) don't have an allocortex but vertebrates
    do have pallium which evolved to cortex in mammals
    -even lamprey and hagfish (ancient jawless fish) have a pallium
    -medial, lateral, dorsal pallium
    -medial pallium is precursor of hippocampus and is homologous in other vertebrates but does
    not look like hippocampus found in mammals
    -evidence that hippocampal-like homologues used for spatial navigation in fish,
    reptiles, and fish -- thus in CCA1 we call all these 'hippocampus' have different
    levels of 'hippocampus' methods
    -insect brain mushroom bodies may have function like hippocampal-like structures in
    vertebrates, but homology uncertain, so we don't deal with in the CCA1
    #
    --documentation of behavior of hippocampus() method
    CURRENTLY BEING TRANSITIONED INTO NANO CODE FROM PREVIOUS MBLS3 MODELS
    '''
    # if no current hippocampal goal direction in memory, then there
    # is not currently a local minimum, and so, reset the variable keeping
    # track how many times we have tried this local minimum
    if not d.h_mem_dirn_goal and d.local_minimum > 0:
        d.local_minimum = 0

    # even given the simple NESW hippocampal goal direction algo below (which may become more complex as code develops)
    # it is possible the cca1 mistakes, eg, an edge for the lost hiker and thus this gets put in the
    # the hippocampal goal direction memory d.h_mem_dirn_goal, and the cca1 endlessly tries to go in this direction
    # thus, first we consider if a local minimum has been reached, and if so, we will simply try to
    # get out of it by going in a random direction
    if d.local_minimum > g.tries_before_declare_local_minimum:
        d.h_mem_prev_dirn_goal = d.h_mem_dirn_goal
        d.h_mem_dirn_goal = None
        direction_to_proceed = hippo2_reasonable_random_direction(d, g, h)
        print(
            "debug: hippo2: local_min reset from {} to 0 and trying random dirn".format(
                d.local_minimum
            )
        )
        return direction_to_proceed

    # update the hippocampal memory, ie, is there any or new goal direction?
    # simply consider NESW surrounding current position and look for hiker
    # (ie, goal) in 1 square distance
    for i, j in ((0, "00"), (1, "01"), (2, "10"), (3, "11")):
        if d.max_fused_index[i] == "0101000011110000":  # lost hiker
            d.h_mem_prev_dirn_goal = d.h_mem_dirn_goal
            d.h_mem_dirn_goal = str(j)
            print(
                "debug: hippo2: h_mem_dirn_goal was just set to {} and local_min is {}".format(
                    d.h_mem_dirn_goal, d.local_minimum
                )
            )

    # is there a value in d.h_mem_dirn_goal, ie, is there a direction the
    # hippocampus has stored as saying we should go in to get the goal?
    if d.h_mem_dirn_goal:
        # increment local minimum tracking variable after a number of tries
        # we do not keep on going in this direction (eg, cca1 may think
        # sensory info indicates hiker in square in this direction but in
        # fact the sensory info is wrong and hiker is not there)
        d.local_minimum += 1
        print(
            "debug: hippo2: d.h_mem_dirn_goal (ie, there is a goal) and local_minimum incremented",
            d.h_mem_dirn_goal,
            d.local_minimum,
        )

        direction = d.h_mem_dirn_goal
        if direction in ["00", 0, "N", "north", "North"]:
            direction = 0
        elif direction in ["01", 1, "E", "east", "East"]:
            direction = 1
        elif direction in ["10", 2, "S", "south", "South"]:
            direction = 2
        elif direction in ["11", 1, "W", "west", "West"]:
            direction = 3
        else:
            print("debug: invalid value for d.h_mem_dirn_goal being evaluated")
            d.h_mem_dirn_goal = None
            direction = 0  # arbitrary assign value for now

        # now check to see if d.h_mem_dirn_goal direction is reasonable
        # for example, new sensory inputs this evaluation cycle may indicate it is not
        print(
            "now evaluating if {} direction from d.h_mem_dirn_goal should be avoided".format(
                d.h_mem_dirn_goal
            )
        )
        # hippo_calc() looks at int_map and decides if should go in a queried direction
        # returns True if ok to go in that direction, False if not ok
        # False if edge, lake
        if d.max_fused_value[direction] == "EDGE  ":  # EDGE
            # g.gconscious(['....but this is thought to be an EDGE thus hippo_calc will be False', direction])
            d.h_mem_dirn_goal = None
            print("not returning d.h_mem_dirn_goal direction as it is an EDGE")
        elif d.max_fused_value[direction] == "lake  ":  # lake
            # g.gconscious(['....but this is thought to be a lake thus try again navigation for this move', direction])
            d.h_mem_dirn_goal = None
            print("not returning d.h_mem_dirn_goal direction as it is a lake")
        else:
            print(
                "debug: d.h_mem_dirn_goal direction {} reasonable".format(
                    d.h_mem_dirn_goal
                )
            )
            return d.h_mem_dirn_goal

    # else d.h_mem_dirn_goal ==None
    # no value in d.h_mem_dirn_goal, thus must decide direction
    direction_to_proceed = hippo2_reasonable_random_direction(d, g, h)
    print(
        "debug: hippo2: trying reasonable random (actually look for hiker) dirn {}".format(
            direction_to_proceed
        )
    )
    return direction_to_proceed


def hippo2_reasonable_random_direction(d, g, h):
    '''if no goal direction to go to then use a reasonable random direction, if not reasonable then try again
    loop where will try random directions and if selected direction is not allowed as per
    the internal map of the cca1, eg, it thinks it is going into an EDGE square, or if the
    selected move is considered non-optimal by hippo_calc2(), then there is continue loop
    and a new random direction is considered
    '''
    # to avoid endlessly retrying random moves that result in non-optimal moves, ie, rejected by hippo_calc2,
    # after have tried MAX_MOVES number of times, do not continue loop again (and get new random direction),
    # but rather return the current direction
    MAX_MOVES = g.tries_before_declare_local_minimum + 3
    no_move_count = 0

    # loop where will try random directions and if selected direction is not allowed as per
    # the internal map of the cca1, eg, it thinks it is going into an EDGE square, or if the
    # selected move is considered non-optimal by hippo_calc2(), then there is continue loop
    # and a new random direction is considered
    while True:
        direction = random.randint(1, 11)
        no_move_count += 1
        print("direction is: ", direction)
        print("no_move_count is: ", no_move_count)
        if no_move_count > MAX_MOVES:
            print(
                "debug: inside hippo2_reasonable_random_direction(): too many attempts to try to find a navigation"
            )
            print("direction, thus random direction arbitrarily chosen")
            if direction in (1, 2):
                return "00"
            if direction in (3, 4, 10, 9):
                return "01"
            if direction in (5, 6, 11):
                return "10"
            if direction in (7, 8):
                return "11"
            print(
                "debug: warning: random value cannot be converted properly to a direction, 00 chosen"
            )
            return "00"  #
        emotional(d.max_fused_index)
        pattern_memory(d.max_fused_index)
        seq_and_error(d.max_fused_index)
        d.h_mem_prev_dirn_goal = d.h_mem_dirn_goal
        d.h_mem_dirn_goal = None
        if direction in (1, 2):
            print("randint direction: attempts to walk northward....")
            if d.max_fused_index[0] == "1111110000000000":  # EDGE
                print("....but this is an EDGE thus try again navigation for this move")
                continue
            if d.max_fused_index[0] == "1110001100010001":  # lake
                print("....but this is a lake thus try again navigation for this move")
                continue
            if not hippo_calc2(d, "00") and no_move_count < MAX_MOVES:
                print(
                    "....but on or low probability of success of goal in this direction"
                )
                continue
            return "00"
        if direction in (3, 4, 10, 9):
            print("randint direction: attempts to walk eastward....")
            if d.max_fused_index[1] == "1111110000000000":  # EDGE
                print("....but this is an EDGE thus try again navigation for this move")
                continue
            if d.max_fused_index[1] == "1110001100010001":  # lake
                print("....but this is a lake thus try again navigation for this move")
                continue
            if not hippo_calc2(d, "01") and no_move_count < MAX_MOVES:
                print(
                    "....but on or low probability of success of goal in this direction"
                )
                continue
            return "01"
        if direction in (5, 6, 11):
            print("randint direction: attempts to walk southward....")
            if d.max_fused_index[2] == "1111110000000000":  # EDGE
                print("....but this is an EDGE thus try again navigation for this move")
                continue
            if d.max_fused_index[2] == "1110001100010001":  # lake
                print("....but this is a lake thus try again navigation for this move")
                continue
            if not hippo_calc2(d, "10") and no_move_count < MAX_MOVES:
                print(
                    "....but on or low probability of success of goal in this direction"
                )
                continue
            return "10"
        if direction in (7, 8):  # prevent westward
            print("randint direction: attempts to walk westward....")
            if d.max_fused_index[3] == "1111110000000000":  # EDGE
                print("....but this is an EDGE thus try again navigation for this move")
                continue
            if d.max_fused_index[3] == "1110001100010001":  # lake
                print("....but this is a lake thus try again navigation for this move")
                continue
            if not hippo_calc2(d, "11") and no_move_count < MAX_MOVES:
                print(
                    "....but on or low probability of success of goal in this direction"
                )
                continue
            return "11"
        if direction == 100:  # will not occur if randint set to prevent
            print("randint direction: this move random pause....")
            output_and_shaping(d, g, h, "00", 0)
        else:
            print(
                "debug: warning: randint direction: coding issue -- please check code for goal find hiker"
            )
            return "00"


def hippo_calc2(d, direction, verbose=0, silent=0):
    '''look at internal hippocampal spatial map int_map and decide if
    should go in this direction
    -parameter is direction are considering
    -returns True if should go in direction or False if not to
    -although output is binary, method is allowed to use random function
    to sometimes return False for paths which while possible are unlikely to
    lead to the goal
    -LAMPREY level algo if all squares in that direction already explored then
    return False
    '''
    # input validity as well as forced outputs
    if direction in ["00", 0, "N", "north", "North"]:
        direction = 0
    elif direction in ["01", 1, "E", "east", "East"]:
        direction = 1
    elif direction in ["10", 2, "S", "south", "South"]:
        direction = 2
    elif direction in ["11", 1, "W", "west", "West"]:
        direction = 3
    elif direction in ["99", 99, "allow"]:
        print("hippo_calc forced to allow directional move due to 99 input")
        # g.gconscious(['hippo_calc forced to allow directional move due to 99 input'])
        return True
    elif direction in ["-1", -1, "prevent"]:
        print("hippo_calc forced to prevent directional move due to -1 input")
        # g.gconscious = (['hippo_calc forced to prevent directional move due to -1 input'])
        return False
    else:
        print(
            "debug: warning: hippo_calc: direction not valid -- return False ",
            direction,
        )
        # g.gconscious(['coding issue -- direction not valid -- return False ', direction, '-1', '99'])
        return False
    if verbose:
        print(
            "CHECKPOINT: in hippocalc, direction is {} and d.h_mem_dirn_goal is {}".format(
                direction, d.h_mem_dirn_goal
            )
        )

    # consider if direction is recommended (returns True) or not (returns False)
    # first make sure immediate direction even possible
    # this has already been done in hippo2_reasonable_random_direction() but left here
    # if code used for other function calls
    if d.max_fused_value[direction] == "EDGE  ":  # EDGE
        # g.gconscious(['....but this is thought to be an EDGE thus hippo_calc will be False', direction])
        if not silent:
            print("....but this is thought to be an EDGE thus hippo_calc will be False")
        return False
    if d.max_fused_value[direction] == "lake  ":  # lake
        if not silent:
            print(
                "....but this is thought to be a lake thus try again navigation for this move"
            )
            # g.gconscious(['....but this is thought to be a lake thus try again navigation for this move', direction])
        return False

    # now explore directions to see if should recommend the direction or not
    # current simple algo is to recommend, ie, True, if unexplored squares
    # (or hiker but note that already been checked for in hippocampus2()
    m = d.cca1_position[0]
    n = d.cca1_position[1]
    if direction == 0:  # N
        # if m == 1 will allow erroneous move to EDGE (will not be allowed by move_cca1
        # later anyway) more realistic since cca1's int_map has synthetic construction
        # of what real world, not the actual real world shown in forest_map
        m = m - 1
    if direction == 1:  # E
        n = n + 1
    if direction == 2:  # S
        m = m + 1
    if direction == 3:  # W
        n = n - 1
    if d.int_map_previous[m][n] == "" or d.int_map_previous[m][n] == "hiker  ":
        print(
            "returning True from hippocalc since unexplored square in direction  ",
            direction,
        )
        # g.gconscious(['returning True from hippocalc since unexplored square in direction '])
        return True
    # thus at this point have considered NESW and none of them are unexplored (or contain hiker) thus
    # unsure what square direction to navigate to, thus return False
    # note in future code or higher level hippocampus can of course have better strategies than this
    # one square look ahead strategy
    return False


def hippo5_strategy(d, g, h):
    '''to better simulate biological evolution causal method begins as exact
       copy of precausal and then modifications placed into it
       to better simulate biological evolution causal method begins as exact
       copy of precausal and then modifications placed into it
       -d.fused_matches signal plus instinct will     causally generate an output
       -sensory + context generating the instinct + instinct representing rules
       == sensory + context + rules about context -->
       -allow better interpretation than just sensory association
       -predict future outcome better
       -plan goal directed result better
        ==> beginnings of causal reasoning

       -examples of data structures being passed
       #d.fused_matches =[{N}, {E}, {S}, {W}]  where, eg,  N = {16bit:[81, 'obstruction'], 16bit:[69, 'lake']}
       #d.current_instinct = '10000000'
       #d.max_fused_index = ['N','E','S','W'] where, eg, N = '1111110000000000'
       #d.max_fused_value = [ [N], [E], [S], [W]  ] where [N] = [81, 'obstruction']
       #d.associative_matches_visual = [ {N}, {E}, {S}, {W} ] where {N} = {'11111100': [88, 'obstruction'], '11100011': [62, 'lake']}
       #d.associative_matches_auditory = [ {N}, {E}, {S}, {W} ] where {N} = {'00000000': [88, 'strange silence'], '11000000': [75, 'forest_noise']}
       usage:
       causal_navigation(d, g, 'CAUSAL (HIPPO 5) MEMORY INTEGRATION')

       ---------------
       #ok to walk puddle, shallow river, grass, trees
       #not ok go into lake -- water will damage articulations
       #if lake subsymbolic -> instinctual -> danger -> change path
       #if shallow river subsymbolic -> instinctual -> continue activities
       #instinctual ->motor activities to accomplish goal
       #*white areas river -> no particular action subsymbolic if not
        previous memory of damage, but in causal more causal action should occur
       #damage leg -> associative memory white areas and danger
       #GOAL_RANDOM_WALK = '00000000'
       #GOAL_PRECAUSAL_FIND_HIKER = '11100011'
       ##move_CCA1(direction='00', steps=0)   <------unreachable code
       ##return DEFAULT_VECTOR
       ----------------
       spatial maps and positioning for CCA1
       returns a direction in which move should be
       #
       d.fused_matches = fused = HLNs_sensory_fusion(sensory_input_visual, sensory_input_auditory, fused_dict)
           sensory_input_visual: eg, ['1110 0011', E, S, W]  (in visual_dict would be 'lake')
           sensory_input_auditory: eg, ['00010000', E, S, W] (in auditory_dict would be 'sm sound')
           fused_dict: eg, {'1110 0011  0001 0000': 'lake+sm snd->lake', OTHER DEFN, .... OTHER DEFN}
         fused: eg, ['1110 0011  0001 0000', E, S, W]
         d.fused_matches is 16bit vectors/dicts in NESW dirns corresponding to fused sensory input
           and best matching fused_ref values
           structure is: [ {N}, {E}, {S}, {W} ]
           substructure of each direction is:  { 16bit_vector:[match_value,feature_text], OTHERS }
           eg,
           [
           {   '1111110000000000': [88, 'obstruction + strange silence -> EDGE'],
               '1110001100010001': [75, 'lake + smooth water sound -> lake']           },
           {   '1100011011000000': [94, 'forest + forest noise -> forest'],
               '1110001100010001': [81, 'lake + smooth water sound -> lake']           },
           {   '1110001100010001': [81, 'lake + smooth water sound -> lake'],
               '1100011011000000': [81, 'forest + forest noise -> forest']             },
           {   '1111110000000000': [100, 'obstruction + strange silence -> EDGE'],
               '1110001100010001': [75, 'lake + smooth water sound -> lake']           }
           ]

    cinstinct
    10000000

    d.max_fused_index
    ['1111110000000000', '1100011011000000', '1100011011000000', '1111110000000000']

    d.max_fused_value
    [[88, 'obstruction + strange silence -> EDGE'], [94, 'forest + forest noise -> forest'], [81, 'forest + forest noise -> forest'], [100, 'obstruction + strange silence -> EDGE']]

    d.associative_matches_visual
    [{'11111100': [88, 'obstruction'], '11100011': [62, 'lake']}, {'11000110': [88, 'forest'], '11100011': [75, 'lake']}, {'11111100': [75, 'obstruction'], '11100011': [62, 'lake']}, {'11111100': [100, 'obstruction'], '11100011': [62, 'lake']}]

    d.associative_matches_auditory
    [{'00000000': [88, 'strange silence'], '00010001': [88, 'smooth water sound']}, {'11000000': [100, 'forest_noise'], '11100000': [88, 'bird mating call']}, {'00010001': [100, 'smooth water sound'], '00000000': [75, 'strange silence']}, {'00000000': [100, 'strange silence'], '11000000': [75, 'forest_noise']}]
       d.current_instinct = current_instinct: eg, '1000 0000'
           in global instinct dict: {'1000 0000': 'forward eat/goal', OTHERS }
       d.max_fused_index: eg, ['1110 0011  0001 0000', E, S, W]
       d.max_fused_value: eg,
       #
       hippocampus 1. Lamprey hippocampal/brain analogue - note: will default to quasi-skewed walk')
       hippocampus 2. Fish hippocampal/telencephalon analogue - note: currently reverts back to lamprey')
       hippocampus 3. Reptile hippocampal/pallium analgoue - note: some precausal features')
       hippocampus 4. Mammalian hippocampus - note: currently reverts back to reptile')
       hippocampus 5. Human hippocampus - note: pending status, some simple causal features')
       hippocampus 6. Superintelligence level 1 - note: currently reverts back to lower human level')
       hippocampus 7. Superintelligence level 2 - note: currently reverts back to lower human level')
       #
       --quick review of what hippocampus in biology does (since otherwise literature confusing for
       non-bio background reader due to terminology):
       -mammals have left and right hippocampi
       -in mammals part of the 'allocortex'=='heterogenetic cortex' versus the 'neocortex'
       (neocortex 6 layers vs. 3-4 cell layers in allocortex; types of allocortex: paleocortex,
       archicortex and transitional (ie, to neocortex) periallocortex)
       olfactory system also part of the allocortex
       -considered part of 'limbic system' (=='paleomammalian cortex' midline structures
        on L and R of thalamus, below temporal lobe; involved in emotion, motivation,
        olfactory sense and memory; making memories affected by limibc system)
        (basic limbic system is really amygdala, mammillary bodies, stria medull, nuc Gudden,
        but also tightly connected to limbic thalamus, cingulate gyrus, hippocampus,
        nucleus accumbens, anterior hypothalamus, ventral tegmental area, raphe nuc,
        hebenular commissure, entorhinal cortex, olfactory bulbs)
       -hippocampus involved in spatial memory, in this CCA1 simulation that is what similarly
       name method does
       -hippcampus also involved in putting together portions of memory throughout whole brain
       -given relation between learning and memory, not surprising to find that hippocampus involved
       in learning; in more advanced hippocampal methods, ie, beyond LAMPREY level
       these indeed are implemented in this method
       -hippocampus needed for consolidation of (certain types) of short-term memories
       into long-term memory, and for spatial navigation
       -above note that in primates hippocampus is in bottom part of medial temporal lobe
       -hippocampus = 'hippocampus proper'==Ammon's horn + dentate gyrus
       -hippocampus in all mammals; animals with better spatial memories are found to have
       larger hippocampal structures
       -in other vertebrates (ie, fish to reptiles) don't have an allocortex but vertebrates
       do have pallium which evolved to cortex in mammals
       -even lamprey and hagfish (ancient jawless fish) have a pallium
       -medial, lateral, dorsal pallium
       -medial pallium is precursor of hippocampus and is homologous in other vertebrates but does
       not look like hippocampus found in mammals
       -evidence that hippocampal-like homologues used for spatial navigation in fish,
       reptiles, and fish -- thus in CCA1 we call all these 'hippocampus' have different
       levels of 'hippocampus' methods
       -insect brain mushroom bodies may have function like hippocampal-like structures in
       vertebrates, but homology uncertain, so we don't deal with in the CCA1
       #
       --documentation of behavior of hippocampus() method
       CURRENTLY BEING TRANSITIONED INTO NANO CODE FROM PREVIOUS MBLS3 MODELS'''
    # if no current hippocampal goal direction in memory, then there
    # is not currently a local minimum, and so, reset the variable keeping
    # track how many times we have tried this local minimum
    if not d.h_mem_dirn_goal and d.local_minimum > 0:
        d.local_minimum = 0

    # even given the simple NESW hippocampal goal direction algo below (which may become more complex as code develops)
    # it is possible the cca1 mistakes, eg, an edge for the lost hiker and thus this gets put in the
    # the hippocampal goal direction memory d.h_mem_dirn_goal, and the cca1 endlessly tries to go in this direction
    # thus, first we consider if a local minimum has been reached, and if so, we will simply try to
    # get out of it by going in a random direction
    if d.local_minimum > g.tries_before_declare_local_minimum:
        d.h_mem_prev_dirn_goal = d.h_mem_dirn_goal
        d.h_mem_dirn_goal = None
        direction_to_proceed = hippo2_reasonable_random_direction(d, g, h)
        print(
            "debug: hippo2: local_min reset from {} to 0 and trying random dirn".format(
                d.local_minimum
            )
        )
        return direction_to_proceed

    # update the hippocampal memory, ie, is there any or new goal direction?
    # simply consider NESW surrounding current position and look for hiker
    # (ie, goal) in 1 square distance
    for i, j in ((0, "00"), (1, "01"), (2, "10"), (3, "11")):
        if d.max_fused_index[i] == "0101000011110000":  # lost hiker
            print("debug: direction {} is lost hiker".format(j))
            d.h_mem_prev_dirn_goal = d.h_mem_dirn_goal
            d.h_mem_dirn_goal = str(j)
            print(
                "debug: hippo2: h_mem_dirn_goal was just set to {} and local_min is {}".format(
                    d.h_mem_dirn_goal, d.local_minimum
                )
            )

    # is there a value in d.h_mem_dirn_goal, ie, is there a direction the
    # hippocampus has stored as saying we should go in to get the goal?
    if d.h_mem_dirn_goal:
        # increment local minimum tracking variable after a number of tries
        # we do not keep on going in this direction (eg, cca1 may think
        # sensory info indicates hiker in square in this direction but in
        # fact the sensory info is wrong and hiker is not there)
        d.local_minimum += 1
        print(
            "debug: hippo2: d.h_mem_dirn_goal (ie, there is a goal) and local_minimum incremented",
            d.h_mem_dirn_goal,
            d.local_minimum,
        )

        direction = d.h_mem_dirn_goal
        if direction in ["00", 0, "N", "north", "North"]:
            direction = 0
        elif direction in ["01", 1, "E", "east", "East"]:
            direction = 1
        elif direction in ["10", 2, "S", "south", "South"]:
            direction = 2
        elif direction in ["11", 1, "W", "west", "West"]:
            direction = 3
        else:
            print("debug: invalid value for d.h_mem_dirn_goal being evaluated")
            d.h_mem_dirn_goal = None
            direction = 0  # arbitrary assign value for now

        # now check to see if d.h_mem_dirn_goal direction is reasonable
        # for example, new sensory inputs this evaluation cycle may indicate it is not
        print(
            "now evaluating if {} direction from d.h_mem_dirn_goal should be avoided".format(
                d.h_mem_dirn_goal
            )
        )
        # hippo_calc() looks at int_map and decides if should go in a queried direction
        # returns True if ok to go in that direction, False if not ok
        # False if edge, lake
        if d.max_fused_value[direction] == "EDGE  ":  # EDGE
            # g.gconscious(['....but this is thought to be an EDGE thus hippo_calc will be False', direction])
            d.h_mem_dirn_goal = None
            print("not returning d.h_mem_dirn_goal direction as it is an EDGE")
        elif d.max_fused_value[direction] == "lake  ":  # lake
            # g.gconscious(['....but this is thought to be a lake thus try again navigation for this move', direction])
            d.h_mem_dirn_goal = None
            print("not returning d.h_mem_dirn_goal direction as it is a lake")
        else:
            print(
                "debug: d.h_mem_dirn_goal direction {} reasonable".format(
                    d.h_mem_dirn_goal
                )
            )
            return d.h_mem_dirn_goal

    # else d.h_mem_dirn_goal ==None
    # no value in d.h_mem_dirn_goal, thus must decide direction
    direction_to_proceed = hippo2_reasonable_random_direction(d, g, h)
    print(
        "debug: hippo2: trying reasonable random (actually look for hiker) dirn {}".format(
            direction_to_proceed
        )
    )
    return direction_to_proceed


def int_map_update(d, direction="00", geo_feature="forest"):
    '''internal hippocampal spatial map constructed and used by the CCA1
    note: in "nano" version the philosophy is to emulate modules which are replaced
     by more authentic components in the finer grain simulations, thus very artificial
     cartesian map constructed here
    #global variable recap:
    d.h_mem_dirn_goal = None
    d.h_mem_prev_dirn_goal = None
    d.cca1_position = (INITIATE_VALUE, INITIATE_VALUE)
    hiker_position = (INITIATE_VALUE, INITIATE_VALUE)
    forest_map = [['forest', 'forest', 'sh_rvr', 'forest'],
                  ['lake  ', 'forest', 'forest', 'forest'],
                  ['forest', 'wtrfall', 'forest', 'forest'],
                  ['forest', 'forest', 'forest', 'forest']]
    nb. m rows x n columns coordinates, start 0,0 --
    forest_map coords superimposed on int_map below
    int_map = [['', '', '', '', '', ''],
                ['', 0,0'', '', '', 0,3'', ''],
                ['', 1,0'', '', '', '', ''],
                ['', 2,0'', '', '', '', ''],
                ['', 3,0'', '', '', 3,3'', ''],
                ['', '', '', '', '', '']]
    nb. start at 0,0 also, note includes EDGE squares which forest_map does not
    #
    see note about this being emulation level in "nano" version
    nonetheless, the function is to provide CCA1 history of where this
    direction, this location leads to
    '''
    # convert d.cca1_position from forest map into int_map coords
    m, n = (d.cca1_position[0] + 0, d.cca1_position[1] + 0)
    # flag that square as being one where CCA1 was already if no geo_feature
    if d.int_map[m][n] == "":
        d.int_map[m][n] = "explored"
    # now flag square to direction specified with geo_feature given
    # nb in this "nano" version just overwrite whatever is there, but in more
    # authentic finer grain simulations to consider validity of data better
    # (geo_feature 'uncertain' used for this purpose as well as expanded set of
    # geo_features)
    if direction in [0, "00"]:
        m = m - 1
    elif direction in [1, "01"]:
        n = n + 1
    elif direction in [2, "10"]:
        m = m + 1
    elif direction in [3, "11"]:
        n = n - 1
    else:
        print(f"debug: warning: int_map_update() -- direction not valid: {direction}")
        input("press any key to continue")
        return False
    if geo_feature not in (
        "hiker",
        "cca1",
        "explored",
        "forest",
        "sh_rvr",
        "lake",
        "wtrfall",
        "EDGE",
        "uncertain",
    ):
        print("warning: coding issue -- geo_feature not valid: ", geo_feature)
        input("press any key to continue....")
        return False
    d.int_map[m][n] = geo_feature
    # g.gconscious([d.int_map])
    return True


def associative_navigation():
    '''associative recognition of sensory inputs,
    some small planning in reaching a goal but at present much is skewed walk
    #
    hippocampus 1. Lamprey hippocampal/brain analogue - note: will default to quasi-skewed walk')
    hippocampus 2. Fish hippocampal/telencephalon analogue - note: currently reverts back to lamprey')
    hippocampus 3. Reptile hippocampal/pallium analgoue - note: some precausal features')
    hippocampus 4. Mammalian hippocampus - note: currently reverts back to reptile')
    hippocampus 5. Human hippocampus - note: pending status, some simple causal features')
    hippocampus 6. Superintelligence level 1 - note: currently reverts back to lower human level')
    hippocampus 7. Superintelligence level 2 - note: currently reverts back to lower human level')
    #
    '''
    direction = random.randint(1, 11)

    # goal, autonomic, instinct
    # planning

    # since N and E are not possible at starting origin, skew random generation
    if direction in (1, 2):
        print("Goal is (skewed) random walk.... so walks northward....")
        return "00"
    if direction in (3, 4, 10, 9):
        print("Goal is (skewed) random walk.... so walks eastward....")
        return "01"
    if direction in (5, 6, 11):
        print("Goal is (skewed) random walk.... so walks southward....")
        return "10"
    if direction in (7, 8):
        print("Goal is (skewed) random walk.... so walks westward....")
        return "11"
    input("debug: warning: non-valid direction in associatve_navigation()")
    return "00"


def output_and_shaping(d, g, h, output_vector, steps_specified=1):
    '''assume 4x4 grid, ie, from 0,0 to 3,3
    calls "move_CCA1(direction, steps)" which returns:
        -if 'forest' square returns 'CCA1  '
        -if 'hiker ' square returns 'RESCUE'
        -if 'lake  ' square returns 'LOSS  '
        -if 'wtrfall' square returns 'DAMAGE'
        -if 'sh_rvr' square returns 'CROSS '
        -if no move then    returns 'STOP  '
    '''
    steps = steps_specified
    if steps == 0:
        move_CCA1(d, "E", 0)
        # return from move_CCA1 will be 'STOP  ' but don't use
    elif output_vector == g.escape_left:
        if d.cca1_position[1] == 1:
            print("CCA1 already all the way left, thus will jump escape right")
            result_of_move = move_CCA1(d, "E", steps)
            if result_of_move == "RESCUE":
                rescue(d, h)
            if result_of_move == "DAMAGE":
                mission_failure(h, "fall damaged an articulation")
            if result_of_move == "CROSS ":
                print("CCA1 ok to cross a shallow river.... all is fine")
            if result_of_move == "LOSS  ":
                mission_failure(h, "walked into a deep lake")
        else:
            print("Escape motion to left occurred")
            result_of_move = move_CCA1(d, "W", steps)
            if result_of_move == "RESCUE":
                rescue(d, h)
            if result_of_move == "DAMAGE":
                mission_failure(h,"fall damaged an articulation")
            if result_of_move == "CROSS ":
                print("CCA1 ok to cross a shallow river.... all is fine")
            if result_of_move == "LOSS  ":
                mission_failure(h, "walked into a deep lake")
    elif output_vector in ("00", 0, "N", "north"):
        # print('n', d.cca1_position)
        if d.cca1_position[0] == 1:
            print("Note: Already northernmost -- no motion this evaln cycle.")
        result_of_move = move_CCA1(d, "00", steps)
        if result_of_move == "RESCUE":
            rescue(d, h)
        if result_of_move == "DAMAGE":
            mission_failure(h, "fall damaged an articulation")
        if result_of_move == "CROSS ":
            print("CCA1 ok to cross a shallow river.... all is fine")
        if result_of_move == "LOSS  ":
            mission_failure(h, "walked into a deep lake")
    elif output_vector in ("01", 1, "E", "east"):
        # print('e', d.cca1_position)
        if d.cca1_position[1] == 4:
            print("Note: Already easternmost -- no motion this evaln cycle.")
        result_of_move = move_CCA1(d, "01", steps)
        if result_of_move == "RESCUE":
            rescue(d, h)
        if result_of_move == "DAMAGE":
            mission_failure(h, "fall damaged an articulation")
        if result_of_move == "CROSS ":
            print("CCA1 ok to cross a shallow river.... all is fine")
        if result_of_move == "LOSS  ":
            mission_failure(h, "walked into a deep lake")
    elif output_vector in ("10", 2, "S", "south"):
        # print('s', d.cca1_position)
        if d.cca1_position[0] == 4:
            print("Note: Already southernmost -- no motion this evaln cycle.")
        result_of_move = move_CCA1(d, "10", steps)
        if result_of_move == "RESCUE":
            rescue(d, h)
        if result_of_move == "DAMAGE":
            mission_failure(h, "fall damaged an articulation")
        if result_of_move == "CROSS ":
            print("CCA1 ok to cross a shallow river.... all is fine")
        if result_of_move == "LOSS  ":
            mission_failure(h, "walked into a deep lake")
    elif output_vector in ("11", 3, "W", "west"):
        # print('w', d.cca1_position)
        if d.cca1_position[1] == 1:
            print("Note: Already westernmost -- no motion this evaln cycle.")
        result_of_move = move_CCA1(d, "11", steps)
        if result_of_move == "RESCUE":
            rescue(d, h)
        if result_of_move == "DAMAGE":
            mission_failure(h, "fall damaged an articulation")
        if result_of_move == "CROSS ":
            print("CCA1 ok to cross a shallow river.... all is fine")
        if result_of_move == "LOSS  ":
            mission_failure(h, "walked into a deep lake")
    else:
        print(
            "coding issue -- please check code for output_vector sent to output_and_shaping: ",
            output_vector,
        )
        input("press any key to continue")
        return False
    return True


def update_expected_values(d, g, h):
    '''in_use_do_not_archive
    called before exit from cycles loop
    -due to the proof of concept stage of the work, a measure of a CCA1 model’s
    future expected value is simply taken as the reciprocal of the number of moves
    required to complete a mission
    d.raw_future_expected_value_for_this_mission is simply the number of moves taken in this mission
    g.raw_future_expected_values_from_multiple_missions simply appends results of each mission
    '''
    #future value update
    print('\nUpdate Expected Values\n')
    d.raw_future_expected_value_for_this_mission = d.evaluation_cycles
    g.raw_future_expected_values_from_multiple_missions.append([d.current_goal, h.exit_reason, d.raw_future_expected_value_for_this_mission])
    print("raw_future_expected_value_for_this_mission since mission started, ie, moves or cycles:  {}\n\n".format(d.raw_future_expected_value_for_this_mission))
    print("raw_future_expected_values_from_multiple_missions: ", g.raw_future_expected_values_from_multiple_missions)
    print("\nfuture_expected value used to adjust behavior and learning")
    print("action taken: None at this time\n")

    #mission complete message
    print("\nScene/mission ended for reason: ", h.exit_reason)
    print("after this number of evaluation cycles:", d.evaluation_cycles)
    g.large_letts_display("Run Complete")
    return d, g, h


def rescue(d, h, secs=1):
    '''called when rescue goal is achieved'''
    beep_secs(secs)
    print("\nA 'rescue' of the lost hiker has occurred.")
    print("When the CCA1 moves to the square of the lost hiker this assumes")
    print("that the CCA1 now follows routines to assist or carry the lost hiker")
    print("back to civilization and medical evaluation.")
    print("\nThis may be the only goal of the system or one of many goals it has.")
    print("At present, even if there are multiple goals, such as an autonomous rest")
    print(
        "goal or perhaps a random walk goal, if a rescue occurs it will be considered"
    )
    print("a valid rescue.")
    print("\nYou can accept the rescue and the program will end, or you can")
    print(" ignore it and continue in the program.")
    aa = input(
        "Accept rescue and end program ('y','yes',ENTER) or no ('n','no') and continue: "
    )
    if aa not in ("n", "N", "no", "No", "NO", "stop", "break"):
        print("Rescue effected. The lost hiker has been returned to civilization,")
        print("undergone a successful medical evaluation and is now enjoying a meal")
        print("with friends and family.")
        print("'Good job CCA1 !!'")
        print("\nProgram will end now (break out of main loop at common point)")
        h.exit_cycles = True  # thus will exit out of cycles loop at loop end
        d.exit_reason = "success"
        return True
    print("Ok...will continue to run....")
    return False


def mission_failure(h, reason="reason not specified"):
    '''called when CCA1 + robot become incapacitated or
    mission fails for other reasons
    '''
    beep_secs(1)
    print("\nA Mission finding the lost hiker has failed.")
    print("Reason: ", reason)
    print("\nYou can stop the rescue and the program will end, or you can")
    print(" ignore it and continue in the program.")
    aa = input(
        "Accept mission stop and end program ('y','yes',ENTER) or no ('n','no'): "
    )
    if aa not in ("n", "N", "no", "No", "NO", "stop", "break"):
        print("Rescue mission stopped. Unfortunately not completed.")
        print("\nProgram will end now (break out of main loop at common point)")
        h.exit_cycles = True  # thus will exit out of cycles loop at loop end
        h.exit_reason = "mission failure"
        return True
    print("Ok...will continue to run....")
    return False


def move_CCA1(d, direction="00", steps=1):
    '''moves CCA1 position on the forest map
    forest_map is m x n coordinate system, start 0,0
    move_CCA1(direction, steps) returns:
    -if 'forest' square returns 'CCA1  '
    -if 'hiker ' square returns 'RESCUE'
    -if 'lake  ' square returns 'LOSS  '
    -if 'wtrfall' square returns 'DAMAGE'
    -if 'sh_rvr' square returns 'CROSS '
    -if no move then    returns 'STOP  '
    '''
    # calculate new position
    steps = int(steps)
    delta_x = 0
    delta_y = 0
    current_x = d.cca1_position[0]
    current_y = d.cca1_position[1]
    if steps < 0:
        print("CCA1 did not move. CCA1 remains in place this evaluation cycle.")
        print("coding issue -- steps should not be negative value")
        d.print_forest_map()
        return "STOP  "
    if steps == 0:
        print("CCA1 did not move. CCA1 remains in place this evaluation cycle.")
        d.print_forest_map()
        return "STOP  "
    # m x n coordinate system, so go down south means go to higher row m
    if direction in ("north", "N", "00"):
        delta_x = -steps
    elif direction in ("south", "S", "10"):
        delta_x = steps
    elif direction in ("east", "E", "01"):
        delta_y = steps
    elif direction in ("west", "W", "11"):
        delta_y = -steps
    else:
        print("coding issue -- direction not valid")
        print(
            "south direction arbitrarily used for number of steps specified or by default"
        )
        delta_y = steps
    current_x += delta_x
    current_y += delta_y
    if current_x < 1:
        current_x = 1
        print(
            "cca1 is attempting to move into an EDGE square which is physically not possible"
        )
        print(
            "feedback to cca1: 0 punishment 0 reward   at this time simply no move occurs"
        )
        return "CCA1  "
    if current_x > 4:
        current_x = 4
        print(
            "cca1 is attempting to move into an EDGE square which is physically not possible"
        )
        print(
            "feedback to cca1: 0 punishment 0 reward   at this time simply no move occurs"
        )
        return "CCA1  "
    if current_y < 1:
        current_y = 1
        print(
            "cca1 is attempting to move into an EDGE square which is physically not possible"
        )
        print(
            "feedback to cca1: 0 punishment 0 reward   at this time simply no move occurs"
        )
        return "CCA1  "
    if current_y > 4:
        current_y = 4
        print(
            "cca1 is attempting to move into an EDGE square which is physically not possible"
        )
        print(
            "feedback to cca1: 0 punishment 0 reward   at this time simply no move occurs"
        )
        return "CCA1  "
    # update d.cca1_position
    print("\nCCA1 moved from {}  {},{}".format(d.cca1_position, current_x, current_y))
    previous_x = d.cca1_position[0]
    previous_y = d.cca1_position[1]
    d.cca1_position = current_x, current_y
    # update forest_map
    # change previous square
    if d.forest_map[previous_x][previous_y] == "RESCUE":
        d.forest_map[previous_x][previous_y] = "hiker "
    elif d.forest_map[previous_x][previous_y] == "DAMAGE":
        d.forest_map[previous_x][previous_y] = "wtrfall"
    elif d.forest_map[previous_x][previous_y] == "CROSS ":
        d.forest_map[previous_x][previous_y] = "sh_rvr"
    elif d.forest_map[previous_x][previous_y] == "LOSS  ":
        d.forest_map[previous_x][previous_y] = "lake  "
    else:
        d.forest_map[previous_x][previous_y] = "forest"
    # change new square
    if d.forest_map[current_x][current_y] == "hiker ":
        d.forest_map[current_x][current_y] = "RESCUE"
        print("\n**CCA1 has rescued lost hiker**")
        d.print_forest_map()
        return "RESCUE"
    if d.forest_map[current_x][current_y] == "RESCUE":
        print("\n**CCA1 has already rescued lost hiker**")
        d.print_forest_map()
        return "RESCUE"
    if d.forest_map[current_x][current_y] == "wtrfall":
        d.forest_map[current_x][current_y] = "DAMAGE"
        print(
            "\n**CCA1 has gone into river leading to a waterfall and was damaged in a fall**"
        )
        d.print_forest_map()
        return "DAMAGE"
    if d.forest_map[current_x][current_y] == "DAMAGE":
        print("\n**CCA1 has already been damaged in the lake**")
        d.print_forest_map()
        return "DAMAGE"
    if d.forest_map[current_x][current_y] == "sh_rvr":
        d.forest_map[current_x][current_y] = "CROSS "
        print(
            "\n**CCA1 has gone into a shallow river which it can successfully cross**"
        )
        d.print_forest_map()
        return "CROSS "
    if d.forest_map[current_x][current_y] == "CROSS ":
        print("\n**CCA1 has successfully crossed a shallow river**")
        d.print_forest_map()
        return "CROSS "
    if d.forest_map[current_x][current_y] == "lake  ":
        d.forest_map[current_x][current_y] = "LOSS  "
        print("\n**CCA1 has gone into a lake and is completely lost**")
        d.print_forest_map()
        return "LOSS  "
    if d.forest_map[current_x][current_y] == "LOSS  ":
        print("\n**CCA1 has already been lost in the lake**")
        d.print_forest_map()
        return "LOSS  "
    d.forest_map[current_x][current_y] = "CCA1  "
    d.print_forest_map()
    return "CCA1  "


'''
hold for now and then delete or deprecate -->
#AUTONOMIC & INSTINCT current_autonomic (ie, energy, damage, etc) and current instinctual state of CCA1
d.current_autonomic = get_current_autonomic(d, g, 'CURRENT AUTONOMIC STATUS MODULE')
#nb d.age_autonomic_calls is incremented here
print('returned d.curent_autonomic : ', d.current_autonomic, ' --> ', d.autonomic_dict[d.current_autonomic])
#input('before current instinct.....')
d.current_instinct = get_current_instinct(d, g, h, 'GET CURRENT INSTINCT')
print('returned d.current_instinct: ', d.current_instinct, ' --> ', d.instinct_dict[d.current_instinct])
#input('after current instinct')


#ASSOCIATIVE, PRE-CAUSAL OR CAUSAL NAVIGATION
print('\n--> NAVIGATION PROCESSING')
print('maxed_fused_index: ', d.max_fused_index, '\nmaxed_fused_value: ', d.max_fused_value)
print('current goal is: ', d.current_goal, '\ncurrent hippocampus is : ', h.current_hippocampus)
if h.current_hippocampus in ['HUMAN_AUGMENTATION', 'SUPERINTELLIGENCE1', 'SUPERINTELLIGENCE2', 'COLLECTIVE_INTELLIGENCE'] or h.current_hippocampus not in ['LAMPREY', 'REPTILE', 'HUMAN']:
    print('debug: warning: h.current_hippocampus {} not supported at this time, and will be set to HUMAN'.format(h.current_hippocampus))
    h.current_hippocampus = 'HUMAN'

if h.current_hippocampus == 'LAMPREY':
    #associative navigation algos to be used
    print('\nASSOCIATIVE (HIPPO 1) MEMORY INTEGRATION')
    if d.associative_matches_visual == 'SPECIAL' or d.associative_matches_auditory == 'SPECIAL':
        print('deprecated')
    navigation_decision = associative_navigation()
    output_and_shaping(d, g, h, navigation_decision, 1)

if h.current_hippocampus == 'REPTILE':
    #precausal navigation algos to be used
    print('\nPRE-CAUSAL (HIPPO 2) MEMORY INTEGRATION')
    update_hippoc_int_map(d)
    navigation_decision = hippocampus2(d, g, h)
    output_and_shaping(d, g, h, navigation_decision, 1)

if h.current_hippocampus == 'HUMAN':
    #causal navigation algos to be used
    print('\nCAUSAL (HIPPO 5) MEMORY INTEGRATION')
    d.current_goal = g.goal_causal_find_hiker
    #hippo5_map(d)
    update_hippoc_int_map(d)
    navigation_decision = hippo5_strategy(d, g)
    output_and_shaping(d, g, h, navigation_decision, 1)
    #examples of data structures being passed
    #d.fused_matches =[{N}, {E}, {S}, {W}]  where, eg,  N = {16bit:[81, 'obstruction'], 16bit:[69, 'lake']}
    #d.current_instinct = '10000000'
    #d.max_fused_index = ['N','E','S','W'] where, eg, N = '1111110000000000'
    #d.max_fused_value = [ [N], [E], [S], [W]  ] where [N] = [81, 'obstruction']
    #d.associative_matches_visual = [ {N}, {E}, {S}, {W} ] where {N} = {'11111100': [88, 'obstruction'], '11100011': [62, 'lake']}
    #d.associative_matches_auditory = [ {N}, {E}, {S}, {W} ] where {N} = {'00000000': [88, 'strange silence'], '11000000': [75, 'forest_noise']}
'''


#
##END OTHER METHODS     END OTHER METHODS


##START PALIMPSEST     START PALIMPSEST
# 3408 lines of deprecated code transferred to
# module palimpsest.py (old lines 2615 - 6023 ver 23)
#
##END PALIMPSEST     END PALIMPSEST
