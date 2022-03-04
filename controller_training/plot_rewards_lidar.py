from six.moves import cPickle as pickle
from matplotlib import pyplot as plt
import numpy as np
import sys

def process_rewards(all_t, all_steps):

    new_t = []
    new_steps_ave = []
    new_steps_min = []
    new_steps_max = []

    cur_t = 1

    increment = 5000

    new_sum = 0
    new_min = 200
    new_max = -1
    num_points = 0

    for i in range(len(all_t)):

        if all_t[i] >= cur_t * increment:

            new_steps_ave.append(float(new_sum) / num_points)
            new_steps_min.append(new_min)
            new_steps_max.append(new_max)
            new_t.append(cur_t * increment)

            new_sum = all_steps[i]
            new_min = all_steps[i]
            new_max = all_steps[i]

            cur_t += 1

            num_points = 1

            continue

        new_sum += all_steps[i]
        num_points += 1

        if all_steps[i] >= new_max:
            new_max = all_steps[i]

        if new_min >= all_steps[i]:
            new_min = all_steps[i]

    return (np.array(new_steps_ave), np.array(new_steps_min), np.array(new_steps_max), np.array(new_t))

def get_stats(filename, track_length):

    all_steps_ave = []
    all_steps_min = []
    all_steps_max = []

    for k in range(5):

        curfilename = filename + '_' + str(k) + '.pickle'

        try:
            f = open(curfilename, 'rb')
        
        except:
            continue

        stats = pickle.load(f)

        all_t = np.array(stats['t'])
        all_steps = np.array(stats['steps'])

        (new_steps_ave, new_steps_min, new_steps_max, new_t) = process_rewards(all_t, all_steps)

        all_steps_ave.append(new_steps_ave)
        all_steps_min.append(new_steps_min)
        all_steps_max.append(new_steps_max)

        all_t = new_t

    steps_ave = np.mean(np.array(all_steps_ave), axis=0)
    steps_min = np.min(np.array(all_steps_min), axis=0)
    steps_max = np.max(np.array(all_steps_max), axis=0)

    steps_ave = 100 * steps_ave / track_length
    steps_min = 100 * steps_min / track_length
    steps_max = 100 * steps_max / track_length

    return (steps_ave, steps_min, steps_max, all_t)

def main(argv):

    folder_name = argv[0]

    filename_right = folder_name + '/stats_lidar_right_turn_64x64'
    filename_sharp_right = folder_name + '/stats_lidar_sharp_right_turn_64x64'
    filename_complex_64 = folder_name + '/stats_lidar_64x64'
    filename_complex_128 = folder_name + '/stats_lidar_128x128'
    filename_complex_256 = folder_name + '/stats_lidar_256x256'

    simple_track_length = 120.0
    complex_track_length = 400.0

    (steps_ave_right, steps_min_right, steps_max_right, all_t) = get_stats(filename_right, simple_track_length)
    (steps_ave_sharp_right, steps_min_sharp_right, steps_max_sharp_right, all_t) = get_stats(filename_sharp_right, simple_track_length)
    (steps_ave_complex_64, steps_min_complex_64, steps_max_complex_64, all_t) = get_stats(filename_complex_64, complex_track_length)
    (steps_ave_complex_128, steps_min_complex_128, steps_max_complex_128, all_t) = get_stats(filename_complex_128, complex_track_length)
    (steps_ave_complex_256, steps_min_complex_256, steps_max_complex_256, all_t) = get_stats(filename_complex_256, complex_track_length)

    comp_ave = (steps_ave_right + steps_ave_sharp_right)/2
    comp_ave = comp_ave[0:int(comp_ave.shape[0]/2)]

    temp_all_t = 2 * all_t
    comp_all_t = temp_all_t[0:int(comp_ave.shape[0])]

    fig = plt.figure(figsize=(12,10))

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    plt.plot(comp_all_t, comp_ave, 'r-', label='Compositional, 64 neurons')

    plt.plot(all_t, steps_ave_complex_64, 'purple', label='Monolithic, 64 neurons')

    plt.plot(all_t, steps_ave_complex_128, 'g-', label='Monolithic, 128 neurons')

    plt.plot(all_t, steps_ave_complex_256, 'yellow', label='Monolithic, 256 neurons')

    plt.ylabel("Percent of track covered until crash", fontsize=26)
    plt.xlabel("Training time steps", fontsize=26)

    plt.legend(loc='lower right', bbox_to_anchor=(0.8,0.6), fontsize=24)

    plt.savefig('rewards_lidar.png')

if __name__ == '__main__':
    main(sys.argv[1:])
