#!/usr/bin/env python
# pylint: disable=line-too-long
'''in_use_do_not_archive
cca4.py

Causal Cognitive Architecture 4 (CCA4)
Sept 2021 rewrite for CSR Manuscript

-- Demonstrate architecture
-- Link to equations in the CSR Manuscript
-- Allow users to run on normal Win or Linux system without GPU
-- Purpose is to show reader what steps the Causal Cognitive Architecture
   is taking, how it is accomplishing them, etc
-- Swap in full code later for CCA3 --> CCA4 transition

Notes:
-please see old code notes for the voluminous changes from mbca versions to cca version to version
-please the following papers for theory behind the cca3 -- it has been removed from the codebase here
 so that actual code does not get overwhelmed with the documentation:
Schneider, H.: The Meaningful-Based Cognitive Architecture Model of Schizophrenia.
    Cognitive Systems Research 59:73-90 (2020).
Schneider, H.: Causal cognitive architecture 1: Integration of connectionist elements into a navigation-based framework.
    Cognitive Systems Research 66:67-81 (2021).
Schneider, H.: Causal Cognitive Architecture 2: A Solution to the Binding Problem, pending
Schneider, H.: Causal Cognitive Architecture 3: A Solution to the Binding Problem, pending


Notes:
-regarding even older deprecation transition notes:
"nano"/"micro"/"milli"/"full" MBCA coarse/fine grain simulations deprecated code left in some areas still
-November 2019 G12/H12 versions MBLS/MBCA being transitioned to Causal Cognitive Architecture 1

#
overview of cca4.py:
    if __name__ == '__main__': main_eval():
      -instantiations of data and method structures g, d, h
      -loop:
        choose species simulation (lamprey to augmented human)
        choose first envrt which sets up instinctive primitives
        main_mech.cycles()
        print_event_log_memory()
        clear memory -- re-instantiation of d, h (g persists between scenes)
        if not run_again(): break loop and end
        -->else loops again for new envrt --^

#
requirements.txt:
    #environment:
    python 3.9 including standard library
    -at this time, not all dependencies will run in other versions, e.g., python 3.10
    -please use or create venv with these exact versions
    -tested in windows terminal but code should optionally bypass windows-specifc os calls if run on other platforms
    -please post platform issues as not tested yet on other platforms

    #python original source code:
    cca4.py   #hyperparameters main_eval() for top level simulation runs
    main_mech.py #cycles() is effective main() of evaluation cycle
    ddata.py  #class MapData --> 'd'
    gdata.py  #class MultipleSessionsData --> 'g'
    hdata.py  #class NavMod --> 'h'
    constants.py #constants only

    #pypi packages:
    pypi.org: fuzzywuzzy #use for cca4.py to avoid need for gpu's, nn
    pypi.org: numpy  #ver 1.19.3 to ensure compatilibity with python 3.9
    pypi.org: colorama, pyfiglet, termcolor #for ascii art printing
    pypi.org: "pip install types-termcolor" or "mypy --install-types" #install stub packages
    #optional -- code will still run without these modules or libraries
    optional: *.jpg #images to display; at present in working directory; to deprecate
    optional: cca3_images folder #images to display, download from specified github
    optional: pypi.org: icecream #for degugging convenience
    optional: pypi.org: python-Levenshtein-01.12.2 #to speed up fuzzywuzzy
    optional: visual c++ #required by python-Levenshtein-01.12.2
    optional: pytorch 1.9
    optional: cuda 11.4

'''

##START PRAGMAS
#
# pylint: disable=line-too-long
#   prefer to take advantage of longer line length of modern monitors, even with multiple windows
# pylint: disable=invalid-name
#   prefer not to use snake_case style for very frequent data structure or small temp variables
# pylint: disable=bare-except
#   prefer in some code areas to catch any exception rising
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
#   prefer to use comfortable number of branches and statements, especially in user menu communication
# pylint: disable=too-many-arguments
#   prefer use pass full set of objects g, d, h, m, c, a to/from some methods
# other style notes:
# ------------------
#  -before each method it is ok to have comments giving a roadmap of where this method is being called from;
#     these were added to hdata methods after the methods were created to aid in the readability of the code
#     in further development work, and found to be helpful as such (can consider putting within the docstring
#     in the future, ie, __doc__ will include, but for now, seem to work well in reading the code)
#
##END PRAGMAS


## START IMPORTS   START IMPORTS
#
##standard imports -- being used by this module
try:
    import logging
    import pdb
    import sys
    import platform
    import os.path
    import random
    # import time
    # import copy
    # from PIL import Image  # type: ignore
except ImportError:
    print("\nprogram will end -- start module of causal cog arch unable to import standard lib module")
    print("please ensure correct version of python can be accessed")
    sys.exit()
##PyPI imports -- being used by this module
try:
    #import numpy as np  # type: ignore
    #  justification: AwesomePython 9.7, L1 code quality
    from icecream import ic  # type: ignore
    ic('remember to disable icecream (here and other modules) for production code')
    #  justification: Awesome rating 7.9
    #  style note: for quick debugging, otherwise logging or create 'verbose' runtime
    import colorama  # type: ignore
    #  justification: AwesomePython 6.7
    import pyfiglet  # type: ignore
    #  justification: AwesomePython 4.4
    import termcolor
    from termcolor import colored
    #  justification: AwesomePython not rated; pypi stable status, 1.1K dependent packages
except ImportError:
    print("\nprogram will end -- start module of the causal cog arch unable to import a PyPI module")
    print("please check requirements.txt and install all required dependencies")
    sys.exit()
##non-PyPI third-party imports -- being used by this module
try:
    pass
    # justification/ Awesome/LibHunt ratings for non-pypi imports: n/a
except ImportError:
    print("program will end -- start module of the causal cog arch unable to import a third-party module")
    print("please check requirements.txt and install all required dependencies")
    sys.exit()
##CCA3 module imports -- being used by this module
try:
    from constants import LIFESPAN, BINDING, SAVE_RECALL_TO_FROM_STORAGE
    import gdata
    import ddata
    import hdata
    import main_mech
    # import eval_micro  #June 2021 deprecated
    # import eval_milli  #June 2021 deprecated
    # import palimpsest  #nb  without GPU will use excessive resources
except ImportError:
    print("program will end -- start module unable to import a causal cognitive architecture module")
    print("please check requirements.txt and install all required dependencies")
    sys.exit()
#
#
##END IMPORTS          END IMPORTS


##START METHODS     START METHODS
#
def welcome(g) -> bool:
    '''in_use_do_not_archive
    CCA3 ver
    print welcome message
    '''
    if g.fastrun:
        return False
    print('''
    CCA3 -- Causal Cognitive Architecture 3 -- Simulation
    CCA3 Demonstration Version with References to Equations of
    manuscript: 'A Solution to the Binding Problem: Causal Cognitive Architecture 3 (CCA3)'
    Pattern recognition via FuzzyWuzzy instead of ANN, thus no GPU required

    Schneider, H.: The Meaningful-Based Cognitive Architecture Model of Schizophrenia.
        Cognitive Systems Research 59:73-90 (2020)
    Schneider, H.: Causal Cognitive Architecture 1 (CCA1): Integration of Connectionist Elements into a
        Navigation-Based Framework. Cognitive Systems Research 66:67-81 (2021)
    Schneider, H.: Causal Cognitive Architecture 2 (CCA2): A Solution to the Binding Problem, BICA*AI 2021, in press
    Schneider, H.: A Solution to the Binding Problem: Causal Cognitive Architecture 3 (CCA3), Cognitive Systems Research, in press
    ''')
    g.fast_input("\nPress ENTER to continue...\n")

    g.large_letts_display("OVERVIEW")
    print('''
    OVERVIEW OF THIS SIMULATION PROGRAM
    -----------------------------------

    1. In this simulation first you will be asked to specify some of the hyperparameters in terms of loosely analogous
    animal equivalents. For example, you can specify a "reptile hippocampal/pallium analogue."

    [Note: Augmented human brain features may or may not be available (depending on version) but are simply for
    development purposes, with no claims of superintelligence or AGI being made.]

    2. The specified brain is then automatically embedded into a robot body. The robot + the CCA3 architecture are
    called "CCA3 robot" or just "CCA3" -- thus, when you see "robot" or "CCA3" think of a robot body being
    controlled by a CCA3 architecture.

    [CCA3 really refers to the architecture controlling the robot body, but for convenience
    we simply call the whole thing the "CCA3" or the "robot" or the "CCA3 robot."]
    [At this time, you do not have any options with regard to the virtual embodiment specifications. Assume a
    generic-like humanoid body with the ability for sensation, locomotion and ability to manipulate objects.]
    [A low-level pyboard version exists in the palimpsest code for interface to a real world embodiment, but
    currently the CCA3 code and libraries need mods for functional compatibility with MicroPython.]

    ''')
    g.fast_input("\nPress ENTER to continue...\n")
    g.large_letts_display("OVERVIEW  2")
    print('''
    3. Then you will be asked to specify the first scene (i.e., really the first environment)
    your newly manufactured robot sees and senses. (Note there can be many sensory scenes one after the other,
    taking place in an environment. For example, the 'PATIENT' environment starts off with a first scene in
    a hospital room with the robot seeing a patient with a walker. A number of sensory scenes occur after that
    first one as the patient asks the robot for a glass of water.)

    4. After a simulation in an environment is over, i.e., your robot succeeded or failed or time ran
    out, your CCA3 robot can move onto the next environment. Usually its brain and memory will remain intact with
    the previous memories. (Continual learning occurs in the core memory systems of the robot -- moving onto a new
    scene and learning new memories will not affect the old ones, as often occurs in traditional neural networks.)

    If for some reason the robot was physically damaged (e.g., simulation where robot was a search and rescue robot)
    it will automatically be repaired when moving onto the next scene.

    Although memory is usually kept intact, you do have the option of having the robot's brain erased of previous learning
    experiences (sometimes useful if you want to try out a scene again without any prior memories). As well, you also
    can choose a different simulation animal analogue (e.g., lamprey to human).

    5. Afer an environment, you can decide at this point if you want to move to another environment (i.e., another simulation
    in that environment), repeat the same environment, or end the program.


    ''')
    g.fast_input("\nPress ENTER to continue...\n")

    # show images related to architecture
    # temporary code and positioning for now; consider captions and driving code from store of images and text
    g.large_letts_display("DIAGRAMS")
    ret_value = g.show_architecture_related("cca3_architecture.jpg", "CCA3 architecture")
    ret_value = g.show_architecture_related("binding_spatial_features.jpg", "spatial binding in CCA3")
    ret_value = g.show_architecture_related("binding_temporal.jpg", "temporal binding in CCA3")
    return ret_value


def choose_simulation(g: gdata.MultipleSessionsData, h: hdata.NavMod, m: hdata.MapFeatures):
    '''in_use_do_not_archive
    CCA3 ver
    Before evaluation cycles of a simulation version start, user can choose which simulation to run.
    We have tried to wrap the hyperparameters in loosely analogous biological equivalents, e.g.,
    specifying you want the features of the fish brain versus a human brain.
    ('Hyperparameters' in the sense they cannot be inferred but specify an architecture we want to evaluate, ie,
    from a Bayesian pov really     a given set of priors we are specifying, but more, also the range of algorithms
    we are specifying to manipulating the priors. Future models  will consider automatic setting of hyperparameters
    but they should be considered static in the current simulation.)
    Note: Augmented human brain features may be available but are simply for development purposes, with
    noclaims of superintelligence, AGI, and so on being made.
    After a scene (i.e., the simulation in the environment) is over, i.e., the CCA3 robot succeeds or perhaps,
    unfortunately, it got damaged for example and failed, the CCA3 robot's body is refurbished as a new robot. However,
    there is the option of keeping its brain intact with the previous memories or refurbishing its brain to a new robot.
    m = hdata.MapFeatures() #m re-initialized between scenes optionally via choose_simulation
    input parameters:
        d, g data and method structures instantiations
    returns:
        h, m since h,m will be modified by this method
    '''
    # display introductory material if first scene
    if g.mission_counter > 1:
        g.large_letts_display("start  envr't\nrun  #  " + str(g.mission_counter), g.mission_counter)
        print(f"new environment {g.mission_counter} is now starting....\n")
    else:
        # print out computing environment and program title/image
        os.system("cls")
        g.large_letts_display("Computing Environment\n")
        computing_evnrt(h)
        input("Press ENTER to continue....")
        os.system("cls")
        try:
            color_letts = ["white", "red", "green", "cyan", "blue", "white", "white", "magenta"][random.randint(0, 7)]
            colorama.init(strip=not sys.stdout.isatty())  # do not use colors if stdout
            termcolor.cprint(pyfiglet.figlet_format("\n   CCA3"), color_letts, attrs=["bold"])
        except:
            print("CCA3")
            print("nb color image did not display\n")

        # print out welcome message
        welcome(g)
        runs_cycles_message(g)
        g.fast_input("\nPress ENTER to start the simulation....")
        g.large_letts_display("run  #  " + str(g.mission_counter), g.mission_counter)
        print(colored('Equations in the CCA3 Binding paper are for one "evaluation cycle"', 'cyan'))
        print(colored('i.e, processing cylcle, or just "cycle"', 'cyan'))
        print(colored('"Runs" refer to a new environment of input sensory scene. Equations are the same regardless of scene.', 'cyan'))
        g.fast_input("\nPress ENTER to continue....\n")
        g.large_letts_display("enter  hyper-\nparameters:")
        g.large_letts_display("brain   type")
        print(colored('Equations in the CCA3 Binding paper assume "Human-like brain"', 'cyan'))


    # print out simulation (ie, hyperparameter) choices
    print('''
    CHOOSE BRAIN SPECIFICATIONS\n
    Please choose type of "hippocampus"/"brain" which, of course, only loosely
    approximates the biological equivalent (you are effectively setting hyperparameters here):
    0. SAME AS LAST ENVIRONMENT, DO NOT ERASE/REFURBISH THE MEMORY
    1. Lamprey-like brain analogue
    2. Fish-like brain
    3. Reptile-like brain
    4. Mammalian-like brain - note: meaningfulness, precausal
    5. Human-like brain - note: meaningfulness plus full causal features
    6. Augmented Human level 1 - simultaneous application of multiple primitives
    7. Augmented Human level 2 - enhanced generative abilities

    ''')
    if g.mission_counter > 1:
        print(f"Previous environment values: hippocampus was {h.current_hippocampus}, and meaningfulness was {h.meaningfulness}.")

    # input choice
    if g.fastrun:
        b_b = 0
    else:
        try:
            b_b = int(input("Please make a selection:"))
        except:
            print("\n**ENTER or nonstandard input**, therefore will default to the previous environment selection.")
            b_b = 0
    if b_b not in range(0, 8):
        print("Default causal human hippocampus selected.")
        b_b = 5

    if b_b == 0:
        # h.current_hippocampus = no change (or if fist environment 'HUMAN')
        if g.mission_counter <= 1:
            print("No previous scenes to retrieve robot from. (No copies kept in local or network storage.)")
            print("Thus, this is actually a brand new robot, rather than a refurbished robot.")
            print("\nWill default at this time to a brain with associative, precausal and some genuine")
            print("robust causal features. Given a mammalian brain, meaningfulness is present.\n")
            h.current_hippocampus = "HUMAN"
            h.meaningfulness = True
        else:
            print("**CCA3 robot body is refurbished but its brain including memory is left unchanged**")
            print("current_hippocampus remains as: ", h.current_hippocampus, " and meaningfulness remains as: ", h.meaningfulness)
        return h, m  # other portions of the code actually modify h so it is returned

    # for other choices, CCA3 brain is refurbished, thus hdata will be re-instantiated
    # ddata is re-instantiated within main_eval loop, while gdata persists between scenes
    h = hdata.NavMod()
    m = hdata.MapFeatures()
    #c = hdata.CognitiveMapFeatures()
    #a = hdata.AugmentedMapFeatures()

    if b_b == 1:
        # h.current_hippocampus = 'LAMPREY'
        print("\nWill default at this time to a quasi-skewed walk.")
        print(
            "Current status is clean functional simulation to allow future versions of the software"
        )
        print("to have more authentic and sophisticated components.\n")
        h.current_hippocampus = "LAMPREY"
        h.meaningfulness = False
        # will default to quasi-skewed walk

    if b_b == 2:
        # h.current_hippocampus = 'FISH' --> 'LAMPREY'
        print("\nWill revert at this time to lamprey pallium analogue.")
        print(
            "Future versions of the software will have more fish functional components."
        )
        print("Note that fish brain does not allow meaningfulness.\n")
        h.current_hippocampus = "LAMPREY"
        h.meaningfulness = False

    if b_b == 3:
        # h.current_hippocampus = 'REPTILE'
        print(
            "\nWill default at this time to simple pallium analogue with some precausal features"
        )
        print("Note that reptilian brain does not allow meaningfulness.\n")
        h.current_hippocampus = "REPTILE"
        h.meaningfulness = False

    if b_b == 4:
        # h.current_hippocampus = 'MAMMAL' --> 'REPTILE'
        print(
            "\nWill revert at this time to reptile pallium analogue. Important evolutionary and"
        )
        print(
            "conceptual advances in the mammalian brain to be put in coming versions of the software."
        )
        print("However, given mammalian brain, meaningfulness is present.\n")
        h.current_hippocampus = "REPTILE"
        h.meaningfulness = True

    if b_b == 5:
        # h.current_hippocampus = 'HUMAN'
        print(
            "\nWill default at this time to a brain with associative, precausal and some genuine"
        )
        print(
            "robust causal features. Given a mammalian brain, meaningfulness is present.\n"
        )
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True

    if b_b == 6:
        # h.current_hippocampus = 'SUPERINTELLIGENCE' --> 'HUMAN'
        print(
            "\nWill default at this time to a simplified human brain with some associative,"
        )
        print(
            "precausal and some genuine causal features. However, enhanced pattern recognition"
        )
        print(
            "abilities as well as enhanced algorithms for logical operations on the navigation maps."
        )
        print(
            "Of importance, there are multiple full navigation modules in this simulation communicating with"
        )
        print(
            "each other, and allowing simultaneous application of multiple primitives, i.e., not just recognition"
        )
        print(
            "and testing of inputs against multiple navigation maps, but full simultaneous processing of"
        )
        print(
            "effectively multiple hypotheses of processing an input. This is for development purposes, and"
        )
        print(
            "no claim of superintelligence is made. Given supra-mammalian brain, meaningfulness is present."
        )
        print(
            "*Superintelligence features not implemented at present. Reverting to human hippocampus.*\n"
        )
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True

    if b_b == 7:
        # h.current_hippocampus = 'SUPERINTELLIGENCE2' --> 'HUMAN'
        print(
            "\nContains the features of human augmented brain level 1. However, massively enhanced generative"
        )
        print(
            "abilities, i.e., statistically is closer to understanding the full joint probability distribution of for"
        )
        print(
            "example the classic p(x,y) and come up with the best solution to complex problems, rather than more"
        )
        print(
            "discriminitive solutions. In the practical sense, this level of brain augmentation"
        )
        print(
            "can invent at machine speed, and find solutions that otherwise would not seem immediately obvious."
        )
        print(
            "However, no claim of superintelligence is made. Given supra-mammalian brain, meaningfulness is present."
        )
        print(
            "*Superintelligence features not implemented at present. Reverting to human hippocampus.*\n"
        )
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True

    if BINDING:
        print("In the related CCA3 binding article, no equations for other species' brain analogues.")
        print("Thus, currently the choice of other species brain unavailable -- human-like brain model will be used.")
        print("h.current_hippocampus = HUMAN, h.meaningfulness = True")
        h.current_hippocampus = "HUMAN"
        h.meaningfulness = True

    g.fast_input("\nPress ENTER to continue...\n")
    # returns h,m since h,m modified by this method
    return h, m


def runs_cycles_message(g):
    '''in_use_do_not_archive
    prints out what is meant by 'runs', 'cycles', 'scenes'
    '''
    g.large_letts_display("runs & cycles")
    print('\nBelow, each simulation run (whether in a PATIENT hospital room environment, in a')
    print('SUDOKO environment, and so on) is displayed as "run #1", "run #2", and so on.')
    print('\nWithin a simulation "run" there are "evaluation cycles" counted starting from cycle 0,')
    print('cycle 1, and so on. When a new simulation run starts again, the evaluation "cycles" (and the')
    print('input sensory "scenes") start counting from zero again, i.e., "cycle 0", "scene 0".')
    print('\nWithin a simulation "run" there are also "scenes" counted starting from scene 0, scene 1,')
    print('and so on. The scenes represent input data from the external world that the CCA3 is')
    print('sensing. They represent "sensory scenes" (i.e., visual, auditory, olfactory, radar, etc')
    print('sensory information) rather than just a visual scene. If the CCA3 is built and running a')
    print('real robot then these scenes are real hardware input signals. However, below in these simulations')
    print('the sensory scenes generally are simulated. Please note that the scene numbers do not have to')
    print('correspond with the evaluation cycle numbers, since several evaluation cycles may be used')
    print('to process a sensory scene.\n')
    print('For example:')
    print('RUN#1 eg, SUDOKO environment')
    print('     evaluation cycle or CYCLE#0  processsing sensory scene SCENE #0 <--scene related to the SUDOKO environment')
    print('     CYCLE#1 processing SCENE#0 <--scene related to the SUDOKO environment')
    print('     CYCLE#2 processing SCENE#1 <--scene related to the SUDOKO environment')
    print('     ....')
    print('     ....')
    print('RUN#2 eg, HOSPITAL environment')
    print('     CYCLE#0  processsing sensory SCENE #0 <--scene related to the HOSPITAL environment')
    print('     CYCLE#1 processing SCENE#0  <--scene related to the HOSPITAL environment')
    print('     ....')
    print('     ....\n\n')
    return True


def choose_starting_scene(d: ddata.MapData, g: gdata.MultipleSessionsData, h: hdata.NavMod)-> ddata.MapData:
    '''in_use_do_not_archive
    CCA3 ver
    Below the user is asked to specify the first scene the newly manufactured robot sees and senses.
    This first scene will retrieve navigation maps and instincitve primitives related to the scene. For the
    remainder of the scene (i.e., until success or fail to reach the goal) causal cognitive embodiment, ie,
    the 'robot' will be in an environment related to this first scene.

    In future versions of the simulation there will be, of course, the ability to switch environments, as happens
    in the real world all the time. However, at present, each scene is in one environment.

    input parameters:
        d, g data and method structures instantiations
    returns:
        #returns d since d is modified by this method
    '''

    # print out the first scene choices
    g.large_letts_display("start  scene")
    print(
        '''
    CHOOSE ENVIRONMENT FIRST SCENE IS TO START IN\n
    The first scene the newly manufactured/refurbished robot sees and senses will retrieve navigation
    maps and instincitve primitives related to the scene's environment.
    For the remainder of the environment (i.e., until success or fail to reach the goal) the causal cognitive
    embodiment, ie, the 'robot' will be in an environment where the scenes are in this environment.
    For example, in the PATIENT environment simulation, the first scene is the robot seeing a patient using
    a walker in a hospital room. The next scene might be the patient asking for a glass of water. However, all
    the scenes are in the hospital room with the patient. When the scenes related to this patient are complete,
    i.e., the simulation in the hospital room (environment PATIENT) is complete, then you are asked again to
    choose another first scene/environment to run the CCA3 robot in. Perhaps you choose an environment where the
    CCA3 plays a game of Sudoku, or perhaps you want to go back to the hospital room and try the previous simulation
    over again.

    In future versions of the simulation there will be, of course, the ability for the CCA3 to switch
    environments on its own, as happens in the real world all the time. However, at present,
    each set of scenes is in one environment.

    Please specify the first scene (environment) the newly manufactured/refurbished robot sees and senses:

    0. Default choice of patient on a walker (ENTER key will also choose)
    1. Looking a Sudoku game sheet
    2. In the middle of an unknown city
    3. Looking at machine filled with gears3
    4. Looking at trees in a forest
    5. Future use
    '''
    )

    print(colored('Equations assume various sensory stimuli being sensed by the CCA3', 'cyan'))
    print(colored('However, since there is not a robot sensing the real world, but', 'cyan'))
    print(colored('a simulation, we must also simulate the sensory stimuli. This is what is', 'cyan'))
    print(colored('being selected here, i.e, simulation of the external world', 'cyan'))

    # input choice selection
    if g.fastrun:
        b_b = 1  #if run with g.fastrun then this is default first_scene
    else:
        try:
            b_b = int(input("Please make a selection:"))
        except:
            print(
                "\n**ENTER or nonstandard input**, therefore default choice selected."
            )
            b_b = 0
    if b_b not in range(0, 6):
        print("**Selection is a nonstandard choice. Thus default choice selected.")
        b_b = 0

    # input choice sets h.first_scene
    if b_b == 0:
        # h.first_scene = default choice 'PATIENT'
        print("Default first_scene has been selected:")
        print("\nCCA3 recognizes a patient on a walker in front of itself.")
        print("This will trigger retrieval of the navigation maps associated with the patient,")
        print("as well as a goal setting to assist such a patient.")
        h.first_scene = "PATIENT"

    if b_b == 1:
        # h.first_scene = 'SUDOKU'
        print("\nCCA3 recognizes a Sudoku game sheet in front of it.")
        print("This will trigger retrieval of the navigation maps associated with sudoku,")
        print("as well as a goal setting to assist playing such a game.")
        h.first_scene = "SUDOKU"

    if b_b == 2:
        # h.first_scene = 'LOST' --> 'PATIENT'
        print("\nCCA3 cannot recognize the environment.")
        print(
            "Not available in this version. Thus switch first scene to recognizing a patient on a walker in front of it."
        )
        h.first_scene = "PATIENT"

    if b_b == 3:
        # h.first_scene = 'GEARS' --> 'PATIENT'
        print(
            "\nCCA3 recognizes the machine in front of it as a broken machine with gears."
        )
        print(
            "Not available in this version. Thus switch first scene to recognizing a patient on a walker in front of it."
        )
        h.first_scene = "PATIENT"

    if b_b == 4:
        ##h.first_scene = 'FOREST' --> 'PATIENT
        print("\nCCA3 recognizes a forest in front of itself.")
        print(
            "This will trigger retrieval of the navigation maps associated with the forest,"
        )
        print("as well as a goal setting to rescue a lost hiker in the forest.")
        print("Not available currently -- to be implemented shortly.")
        print(
            "Thus switch first scene to recognizing a patient on a walker in front of it."
        )
        h.first_scene = "PATIENT"

    if b_b == 5:
        # h.first_scene = 'NOT_SPECIFIED' --> 'PATIENT'
        print("\nNot specified. Future use..")
        print(
            "Not available in this version. Thus switch first scene to recognizing a patient on a walker in front of it."
        )
        h.first_scene = "PATIENT"

    if (BINDING and b_b != 0):
        print("\nFor the moment, the CCA3 controlling a robot which acts as a patient-aide")
        print("is being developed. Thus, default first_scene has been selected:")
        print("\nCCA3 recognizes a patient on a walker in front of itself.")
        h.first_scene = "PATIENT"
        d.current_goal = g.goal_default

    g.fast_input("\nPress ENTER to continue...\n")
    # returns d since d is modified by this method
    return d


def print_event_log_memory(g: gdata.MultipleSessionsData) -> bool:
    '''in_use_do_not_archive
    CCA3 ver
    print out raw event_log memory for now
    add more functionality in future versions via
    other methods inside the appropriate module
    '''
    if g.fastrun:
        return True
    if input("Print out raw event_log memory?") in ("Y", "y", "Yes", "yes"):
        g.printout_event_log_memory()
        return True
    return False


def recall_from_storage(g, d, h, m, c, a):
    '''in_use_do_not_archive
    CCA4 ver
    recalls values of g, d, h, m, c, a from long term storage media
    '''
    print("recalls values of g, d, h, m, c, a from long term storage media")
    print("long-term storage media: ")
    print("long term storage not available at present\n")
    return g, d, h, m, c, a


def save_to_storage(g, d, h, m, c, a):
    '''in_use_do_not_archive
    CCA4 ver
    saves values of g, d, h, m, c, a to long term storage media
    '''
    print("saves values of g, d, h, m, c, a to long term storage media")
    print("long-term storage media: ")
    print("long term storage not available at present\n")
    return g, d, h, m, c, a


def run_again() -> bool:
    '''in_use_do_not_archive
    CCA3 ver
    check what action to take at end of a scene, ie, run again?
    '''
    if input("\nRun again?") in ("N", "n", "NO", "No", "nO", "N0", "no", "0", "stop", "break"):
        return False
    return True


def start_run_messages(d, g, h):
    '''in_use_do_not_archive
    messages to user and any other preliminary operations
    before a simulation run
    '''
    print("\n----------\nvalues for software development usage:\nh.meaningfulness, h.current_hippocampus, h.first_scene, d.current_goal: ")
    print(h.meaningfulness, h.current_hippocampus, h.first_scene, d.current_goal, "\n----------\n")
    print("\nSTART EVALUATION CYCLES")
    print("(nb. Each 'evaluation cycle' is one loop through the CCA3 architecture.")
    print("Sometimes a new scene will occur after an 'evaluation cycle', sometimes after a few cycles.")
    print("Recall that the 'cycle' is a cycle of processing through the architecture of the sensory scene")
    print("being presented to the CCA3 architecture. A number of processing cycles may occur for a")
    print("particular sensory scene.  'cycle' is internal processing, 'scene' is the external sensory")
    print("stimuli being presented (or simulated) to the CCA3.)\n")
    print(colored('The equations in the CCA3 Binding paper cover only one "cycle"', 'cyan'))
    print(colored('In the next "cycle" the equations largely repeat, although not re-initialized\n', 'cyan'))
    g.fast_input(f"Press ENTER to start the CCA3 evaluation cycles for this environment {h.first_scene} (simulation run # {g.mission_counter} since program started) ....")
    return True


def exit_program(g) -> None:
    '''in_use_do_not_archive
    CCA3 ver
    orderly shutdown of program
    "nano" version no intermediate PyTorch structures to save -- deprecated
    '''
    print("\nOrderly shutdown of program via exit_program()")
    print(
        "Please ignore any messages now generated by main/pyboard/etc detection code...."
    )
    g.large_letts_display("program exit")
    sys.exit()


def computing_evnrt(h) -> bool:
    '''in_use_do_not_archive
    CCA4 ver
    displays information about the computing environment
    '''
    print(colored("** PLEASE MAKE SURE YOUR TERMINAL DISPLAY IS FULL SIZE WITH APPROPRIATE FONT, SIZE 20 **", 'red'))
    print("(Windows terminal - right click on the menu bar, left click on 'Properties', click 'Font', 'Size' == 20, 'Font' == Consolas)")
    print("(Consolas font is 9px wide, 20 px high; click 'Colors', 'Screen Text' == dark green, 'Screen Background' == black)")
    print("(Mac, Linux platforms - please similarly adjust your terminal properties, as needed)")
    print("\n\nInformation about computing environment:")
    print("CCA3 - CCA4 Transition Sept 2021 Version")
    print("(Note: Should bypass any Windows-dependent calls if run on another platform.)")
    try:
        print("CCA4 Project: Python installed: ", os.path.dirname(sys.executable))
        print("Platform Info (via StdLib): \n  ", "Python version: ", sys.version, "\n   os.name:",
            os.name, platform.system(), platform.release(), "sys.platform:", sys.platform, "\n  ",
            "(Windows note: sys.platform may give 'win32' result even if win64 for backwards compatibility reasons)\n",
            "  platform.processor:", platform.processor(), "\n  ",
            "sys.maxsize (9223372036854775807 for 64 bit Python): ", sys.maxsize)
        print("   total navigation maps (i.e., cortical mini-column analogues) available via constants.py: ",
            h.total_maps)
        if BINDING:
            print('For this CCA3 demonstration version no GPUs or cloud software required. No GPU checking.\n\n')
        else:
            try:
                # GPU appropriate library required
                #print("GPU Pytorch CUDA availability: ", torch.cuda.is_available())
                print("Pytorch, CUDA, GPU checking not installed at present")
            except:
                print("Unable to check correctly if GPU_ENABLED")
            print("\n\n")
        return True
    except:
        print("Unable to obtain full computing envrt information\n")
        return False


def embedded_main_pyboard(g) -> None:
    '''in_use_do_not_archive
    CCA3 ver
    check palimpsest for embedded_main_pyboard() code
    intended to allow interface between the causal cognitive architecure and a robot embodiment
    '''
    print("'embedded_main_pyboard()' is currently part of deprecated code")
    input("Program will now be ended.... click any key to continue....")
    exit_program(g)


#
##END METHODS     END METHODS


##START INTRO-MAIN     START INTRO-MAIN
#
def main_eval() -> None:
    '''in_use_do_not_archive
    overview:
        if __name__ == '__main__': main_eval():
          -instantiations of data and method structures g, d, h
          - loop:
            choose species simulation (lamprey to augmented human)
            choose envr't which sets up instinctive primitives
            main_mech.cycles()
                -sensory scenes feeding into the cca3 architecture
                -evaluation cycles occur to process each sensory scene
                -when no more scenes to feed in or other end of simulation run,
                then exit from evaluation cycles
            print_event_log_memory()
            clear memory -- re-instantiation of d, h (g persists between scenes)
            if not run_again(): break loop and end
            -->else loops again for new scene envr't^

    '''
    # set up
    g = gdata.MultipleSessionsData()    #persists between runs
    d = ddata.MapData()                 #re-initialized every run
    h = hdata.NavMod()                  #optional re-initialized each run if no choose '0 Same as Last Brain'
    m = hdata.MapFeatures()             #optional re-initialized each run if no choose '0 Same as Last Brain'
    c = hdata.CognitiveMapFeatures()    #optional re-initialized each run if no choose '0 Same as Last Brain'
    a = hdata.AugmentedMapFeatures()    #optional re-initialized each run if no choose '0 Same as Last Brain'
    if SAVE_RECALL_TO_FROM_STORAGE:
        g, d, h, m, c, a = recall_from_storage(g, d, h, m, c, a)
    #input('\ndebug:view startup messages prior to cls... press ENTER to continue....')
    g.one_moment_please_display(1)
    g.choose_if_g_fastrun_on_off() #set verbosity for devp't

    # siml'n run for a given envr't, then repeat for a new envr't or exit
    for g.mission_counter in range(1, LIFESPAN):  #10,000
        # set up data and hyperparameters for the scene
        print(colored("\n\n\nCCA3 Binding paper software walk-through note:", 'blue'))
        print(colored("main_eval() loop: obtain hyperparameters\n\n", 'blue'))
        g.fast_input("Press ENTER to continue...\n")
        h, m = choose_simulation(g, h, m)
        d = choose_starting_scene(d, g, h)
        start_run_messages(d, g, h)

        # start simulation run of evaluation cycles for the envr't
        print(colored("\n\n\nCCA3 Binding paper software walk-through note:", 'blue'))
        print(colored("main_eval() loop: call main_mech.cycles()\n\n", 'blue'))
        g.fast_input("Press ENTER to continue...\n")
        d, g, h, m = main_mech.cycles(d, g, h, m)

        # return from a simulation run
        print(colored("\n\n\nCCA3 Binding paper software walk-through note:", 'blue'))
        print(colored("main_eval() loop: returned from simulation run\n\n", 'blue'))
        g.fast_input("Press ENTER to continue...\n")
        print_event_log_memory(g)
        if not run_again():
            break
        d = ddata.MapData()  # re-initialize for next simulation run
        # if not exited, then select new (or same) envr't and repeats now again ----^

    # end program
    if SAVE_RECALL_TO_FROM_STORAGE:
        g, d, h, m, c, a = save_to_storage(g, d, h, m, c, a)
    exit_program(g)
#
##END INTRO-MAIN      END INTRO-MAIN


if __name__ == "__main__":
    main_eval()
else:
    print("\n\n\n\nModule ", __name__, " is not named as __main__, thus pyboard version of main being called\n")
    logging.warning('wrong main branch given unavailability of pyboard hardware')
    pyboard_instantiation_g = gdata.MultipleSessionsData()
    embedded_main_pyboard(pyboard_instantiation_g)
pdb.set_trace()


#
##START PALIMPSEST     START PALIMPSEST
# 3408 lines of deprecated code transferred to
# module palimpsest.py (old lines 2615 - 6023 ver 23)
# Feb 2021 -- should not need any of this code at this point
# Feb 2021 -- several thousand lines of other code also cleared out, see prev versions if needed
##END PALIMPSEST     END PALIMPSEST
