from Car import World
from Car import complex_track
from Car import square_hall_right
from Car import square_hall_left
from Car import trapezoid_hall_sharp_right
from Car import triangle_hall_sharp_right
from Car import triangle_hall_equilateral_right
import numpy as np
import random
from keras import models
import matplotlib.pyplot as plt
import sys
import yaml

def predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 0
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = sigmoid(curNeurons)
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return curNeurons    

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def reverse_state(data):
    new_data = np.array([data[1], data[0], data[3], data[2], -data[4]])

    return new_data

def main(argv):

    right_input_filename = 'tanh16x16_right_turn_simple.yml'
    sharp_right_input_filename = 'tanh16x16_sharp_turn_simple.yml'
    
    if 'yml' in right_input_filename:
        with open(right_input_filename, 'rb') as f:
            right_model = yaml.load(f)
    else:
        right_model = models.load_model(right_input_filename)

    if 'yml' in sharp_right_input_filename:
        with open(sharp_right_input_filename, 'rb') as f:
            sharp_right_model = yaml.load(f)
    else:
        sharp_right_model = models.load_model(sharp_right_input_filename)        

    numTrajectories = 100
    
    (hallWidths, hallLengths, turns) = complex_track(1.5)
    
    car_dist_s = hallWidths[0]/2.0
    car_dist_f = 8
    car_heading = 0
    car_V = 2.4
    episode_length = 500
    time_step = 0.1

    state_feedback = True

    lidar_field_of_view = 115
    lidar_num_rays = 21

    lidar_noise = 0

    missing_lidar_rays = 0

    num_unsafe = 0

    w = World(hallWidths, hallLengths, turns,\
              car_dist_s, car_dist_f, car_heading, car_V,\
              episode_length, time_step, lidar_field_of_view,\
              lidar_num_rays, lidar_noise, missing_lidar_rays, True, state_feedback=state_feedback)

    throttle = 16
    action_scale = float(w.action_space.high[0])
    
    allX = []
    allY = []
    allR = []

    # initial uncertainty
    init_pos_noise = 0.1
    init_heading_noise = 0.02

    for step in range(numTrajectories):

        observation = w.reset(pos_noise = init_pos_noise, heading_noise = init_heading_noise)

        init_cond = [w.car_dist_f, w.car_dist_s, w.car_heading, w.car_V]

        rew = 0

        for e in range(episode_length):
            
            if not state_feedback:
                observation = normalize(observation)

            # 90-degree right turn
            if w.turns[w.curHall] == -np.pi / 2:

                # in straight segment
                if w.car_dist_f > 6.5:

                    if 'yml' in right_input_filename:
                        delta = action_scale * predict(right_model, observation.reshape(1,len(observation)))
                    else:
                        delta = action_scale * right_model.predict(observation.reshape(1,len(observation)))
                else:
                    
                    if 'yml' in right_input_filename:
                        delta = action_scale * predict(right_model, observation.reshape(1,len(observation)))
                    else:
                        delta = action_scale * right_model.predict(observation.reshape(1,len(observation)))

            # 90-degree left turn
            elif w.turns[w.curHall] == np.pi / 2:

                # in straight segment
                if w.car_dist_f > 6.5:

                    if 'yml' in right_input_filename:
                        delta = action_scale * predict(right_model, observation.reshape(1,len(observation)))
                    else:
                        delta = action_scale * right_model.predict(observation.reshape(1,len(observation)))
                else:

                    observation = reverse_state(observation)
                    
                    if 'yml' in right_input_filename:
                        delta = -action_scale * predict(right_model, observation.reshape(1,len(observation)))
                    else:
                        delta = -action_scale * right_model.predict(observation.reshape(1,len(observation)))

            # 120-degree right turn
            if w.turns[w.curHall] == - 2 * np.pi / 3:

                # in straight segment
                if w.car_dist_f > 8:

                    if 'yml' in right_input_filename:
                        delta = action_scale * predict(right_model, observation.reshape(1,len(observation)))
                    else:
                        delta = action_scale * right_model.predict(observation.reshape(1,len(observation)))
                else:
                    
                    if 'yml' in right_input_filename:
                        delta = action_scale * predict(sharp_right_model, observation.reshape(1,len(observation)))
                    else:
                        delta = action_scale * sharp_right_model.predict(observation.reshape(1,len(observation)))

            # 120-degree left turn
            if w.turns[w.curHall] == 2 * np.pi / 3:

                # in straight segment
                if w.car_dist_f > 8:

                    if 'yml' in right_input_filename:
                        delta = action_scale * predict(right_model, observation.reshape(1,len(observation)))
                    else:
                        delta = action_scale * right_model.predict(observation.reshape(1,len(observation)))

                else:

                    observation = reverse_state(observation)
                    
                    if 'yml' in right_input_filename:
                        delta = -action_scale * predict(sharp_right_model, observation.reshape(1,len(observation)))
                    else:
                        delta = -action_scale * sharp_right_model.predict(observation.reshape(1,len(observation)))

            observation, reward, done, info = w.step(delta, throttle)

            if done:
                
                if e < episode_length - 1:
                    num_unsafe += 1
                
                break

            rew += reward

        allX.append(w.allX)
        allY.append(w.allY)
        allR.append(rew)

    print('number of crashes: ' + str(num_unsafe))
    
    fig = plt.figure(figsize=(12,10))
    w.plotHalls()
    
    plt.tick_params(labelsize=20)

    for i in range(numTrajectories):
        plt.plot(allX[i], allY[i], 'r-')

    plt.savefig('trajectories_simple.png')
    
if __name__ == '__main__':
    main(sys.argv[1:])
