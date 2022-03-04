from six.moves import cPickle as pickle
import os
import sys
import yaml
import numpy as np
import time

import subprocess
from subprocess import PIPE

from sharp_left_turn import writeComposedSystem
from sharp_left_turn import getCornerDist

from sharp_left_turn import POS_X_LB
from sharp_left_turn import POS_X_UB
from sharp_left_turn import POS_Y_LB
from sharp_left_turn import POS_Y_UB
from sharp_left_turn import HEADING_LB
from sharp_left_turn import HEADING_UB
from sharp_left_turn import EXIT_POS_X_LB
from sharp_left_turn import EXIT_POS_X_UB
from sharp_left_turn import EXIT_HEADING_LB
from sharp_left_turn import EXIT_HEADING_UB
from sharp_left_turn import TURN_ANGLE
from sharp_left_turn import EXIT_DISTANCE
from sharp_left_turn import NUM_STEPS
from sharp_left_turn import UNSAFE_Y

HALLWAY_WIDTH = 1.5
HALLWAY_LENGTH = 20
WALL_LIMIT = 0.15
WALL_MIN = str(WALL_LIMIT)
WALL_MAX = str(1.5 - WALL_LIMIT)
SPEED_EPSILON = 1e-8

TIME_STEP = 0.1

CORNER_ANGLE = np.pi - np.abs(TURN_ANGLE)
SIN_CORNER = np.sin(CORNER_ANGLE)
COS_CORNER = np.cos(CORNER_ANGLE)

name = 'right_turn_'

# just a check to avoid numerical error
if TURN_ANGLE == -np.pi/2:
    name = 'right_turn_'
    SIN_CORNER = 1
    COS_CORNER = 0

NORMAL_TO_TOP_WALL = [SIN_CORNER, -COS_CORNER]


def main(argv):

    numRays = 21

    plantPickle = '../plant_models/dynamics_' + name + '{}.pickle'.format(numRays)
    gluePickle = '../plant_models/glue_{}.pickle'.format(numRays)

    with open(plantPickle, 'rb') as f:

        plant = pickle.load(f)

    with open(gluePickle, 'rb') as f:

        glue = pickle.load(f)

    WALL_MIN = str(WALL_LIMIT)
    WALL_MAX = str(HALLWAY_WIDTH - WALL_LIMIT)

    wall_dist = getCornerDist()

    # F1/10 Safety + Reachability
    safetyProps = 'unsafe\n{\tleft_wallm2000001\n\t{\n\t\ty1 <= ' + WALL_MIN + '\n\n\t}\n' \
        + '\tright_bottom_wallm3000001\n\t{'\
        + '\n\t\ty1 >= ' + WALL_MAX + '\n\t\ty2 >= ' + str(wall_dist - WALL_LIMIT) + '\n\n\t}\n' \
        + '\ttop_wallm4000001\n\t{\n\t\t ' + str(NORMAL_TO_TOP_WALL[0]) + ' * y2 + ' \
        + str(NORMAL_TO_TOP_WALL[1]) + ' * y1 <= ' + WALL_MIN + '\n\n\t}\n' \
        + '\t_cont_m2\n\t{\n\t\tk >= ' + str(NUM_STEPS-1) + '\n\n\t}\n' \
        + '\tm_end_pl\n\t{\n\t\ty1 <= ' + str(EXIT_POS_X_LB) + '\n\n\t}\n' \
        + '\tm_end_pr\n\t{\n\t\ty1 >= ' + str(EXIT_POS_X_UB) + '\n\n\t}\n' \
        + '\tm_end_hl\n\t{\n\t\ty4 >= ' + str(EXIT_HEADING_UB) + '\n\n\t}\n' \
        + '\tm_end_hr\n\t{\n\t\ty4 <= ' + str(EXIT_HEADING_LB) + '\n\n\t}\n' \
        + '\tm_end_sr\n\t{\n\t\ty3 >= ' + str(2.4 + SPEED_EPSILON) + '\n\n\t}\n' \
        + '\tm_end_sl\n\t{\n\t\ty3 <= ' + str(2.4 - SPEED_EPSILON) + '\n\n\t}\n' \
        + '\tm_end_yh\n\t{\n\t\ty2 <= ' + str(EXIT_DISTANCE - UNSAFE_Y) + '\n\n\t}\n' \
        + '\tm_end_mh\n\t{\n\t\tsegment_mode >= 1\n\n\t}\n}'

    modelFolder = '../flowstar_models'
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    modelFile = modelFolder + '/testModel'

    with open(plantPickle, 'rb') as f:

        plant = pickle.load(f)

    with open(gluePickle, 'rb') as f:

        glue = pickle.load(f)

    curLBPos = POS_X_LB
    posOffset = 0.005

    cur_init_y2 = POS_Y_LB
    posOffset_y2 = 0.050

    heading_LB = HEADING_LB
    headingOffset = 0.01

    count = 1

    big = '../../simulator/big.yml'
    straight_little = '../../simulator/straight_little.yml'
    square_right_little = '../../simulator/square_right_little.yml'
    square_left_little = '../../simulator/square_left_little.yml'
    sharp_right_little = '../../simulator/sharp_right_little.yml'
    sharp_left_little = '../../simulator/sharp_left_little.yml'
    square_controller = '../../simulator/tanh64x64_right_turn_lidar.yml'
    sharp_controller = '../../simulator/tanh64x64_sharp_turn_lidar.yml'

    with open(square_controller, 'rb') as f:

        dnn = yaml.load(f)

    num_cores = 60

    all_args = {}

    num_x_instances = round((POS_X_UB - POS_X_LB) / float(posOffset))
    num_y_instances = round((POS_Y_UB - POS_Y_LB) / float(posOffset_y2))
    num_theta_instances = round((HEADING_UB - HEADING_LB) / float(headingOffset))

    x_ind = 0
    y_ind = 0
    theta_ind = 0

    while x_ind < num_x_instances:

        while y_ind < num_y_instances:

            while theta_ind < num_theta_instances:

                initProps = ['y1 in [' + str(curLBPos) + ', ' + str(curLBPos + posOffset) + ']',
                             'y2 in [' + str(cur_init_y2) + ', ' +
                             str(cur_init_y2+posOffset_y2) + ']',
                             'y3 in [' + str(2.4 - SPEED_EPSILON) + ', ' +
                             str(2.4 + SPEED_EPSILON) + ']',
                             'y4 in [' + str(heading_LB) + ', ' + str(heading_LB +
                                                                      headingOffset) + ']',
                             'k in [0, 0]',
                             'u in [0, 0]', 'angle in [0, 0]', 'temp1 in [0, 0]', 'temp2 in [0, 0]',
                             'theta_l in [0, 0]', 'theta_r in [0, 0]', 'ax in [0, 0]',
                             'segment_mode in [0, 0]']  # F1/10

                curModelFile = modelFile + '_' + str(count) + '.model'

                writeComposedSystem(curModelFile, initProps, dnn,
                                    plant, glue, safetyProps, NUM_STEPS, printing='off')

                dnn_string = big + ' ' + straight_little + ' ' + square_right_little + ' ' \
                    + square_left_little + ' ' + sharp_right_little + ' ' + sharp_left_little \
                    + ' ' + square_controller + ' ' + sharp_controller

                args = '../../verisig/flowstar/flowstar ' + dnn_string + ' < ' + curModelFile

                all_args[count] = args

                count += 1
                theta_ind += 1
                heading_LB += headingOffset

            cur_init_y2 += posOffset_y2
            y_ind += 1
            theta_ind = 0
            heading_LB = HEADING_LB

        curLBPos += posOffset
        cur_init_y2 = POS_Y_LB
        x_ind += 1
        y_ind = 0

    cur_process = 1
    cur_running = {}

    while cur_process <= num_cores:

        if cur_process not in all_args:
            break

        p = subprocess.Popen(all_args[cur_process], shell=True, stdin=PIPE)

        cur_running[cur_process] = p

        cur_process += 1

    while cur_process <= len(all_args):

        time.sleep(30)

        processes_to_del = []
        processes_to_add = {}

        # first poll
        for process_id in cur_running:

            poll = cur_running[process_id].poll()

            if poll is None:
                pass
            else:
                processes_to_del.append(process_id)

                p = subprocess.Popen(all_args[cur_process], shell=True, stdin=PIPE)

                print(' started process ' + str(cur_process))

                processes_to_add[cur_process] = p

                cur_process += 1

        # delete finished processes
        for process_id in processes_to_del:
            del cur_running[process_id]

        # add new processes
        for process_id in processes_to_add:
            cur_running[process_id] = processes_to_add[process_id]


if __name__ == '__main__':
    main(sys.argv[1:])
