import importlib

def check_and_install(package):
    try:
        importlib.import_module(package)
        print(f'{package} is already installed.')
    except ImportError:
        print(f'{package} is not installed. Installing...')
        import subprocess
        subprocess.check_call(['pip', 'install', package])
        print(f'{package} has been installed successfully.')


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib import animation
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython import display
from tqdm import tqdm
import random
import plotly.express as px
import seaborn as sns
import argparse
from datetime import timedelta
from sklearn.model_selection import train_test_split
import os
from dataloader import Data2Numpy
def plot_trajectory(path, tag_id = None, trajectory_name = None, annotate = False):
    # Converting to datetime object
    data = pd.read_csv(path, sep=";")
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    
    # data for tag_id on that date
    if tag_id is not None:
        data = data[data['tag_id'] == tag_id]
    if trajectory_name is not None:
        data = data[data['trajectory_name'] == trajectory_name] 
    filtered_data = data

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(24, 12))
    ax.scatter(filtered_data['x'], filtered_data['y'])  # Plot the coordinate points
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_title(f'Trajectory for tag {tag_id}')

    # Iterate over the coordinates
    for i, coord in enumerate(zip(filtered_data['x'], filtered_data['y'])):
        x, y = coord
        ax.plot(x, y, 'bo')  # Plot the coordinate point
        if annotate:
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,10), ha='center')  # Annotate with the index

        if i > 0:
            prev_x, prev_y = filtered_data.iloc[i-1]['x'], filtered_data.iloc[i-1]['y']
            ax.plot([prev_x, x], [prev_y, y], 'b-')  # Connect previous coordinate with current coordinate using a line

    # Display the plot


    # Set x-axis tick interval to 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # Set y-axis tick interval to 1
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def traj_gif(tag_id, trajectory_name, path):
    data = pd.read_csv(path, sep=';')
    filtered_data = data[(data['tag_id'] == tag_id) & (data['trajectory_name'] == trajectory_name)]
    points = [(x,y) for x, y in zip(filtered_data['x'], filtered_data['y'])]
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot(x[0:1], y[0:1], color = 'grey' )
    dot, = ax.plot(x[0], y[0], color = 'black', marker = 'o' )

    ax.set_xlim([0,45])
    ax.set_xlabel('x', fontsize = 14)
    ax.set_ylim([0,45])
    ax.set_ylabel('y', fontsize = 14)
    ax.set_title(f'Relationship between x and y at step 0', fontsize=14)

    def update_frame(t):
        line.set_data(x[0:t+1], y[0:t+1])
        dot.set_data(x[t], y[t])
        ax.set_title(f'Relationship between x and y at step {t}', fontsize=14)
        return line, dot

    time = np.arange(len(x))
    anim = animation.FuncAnimation(fig, update_frame, frames=time, interval=500)
    anim.save(f'{tag_id}_{trajectory_name}.gif', writer='pillow')

    with open(f'{tag_id}_{trajectory_name}.gif','rb') as f:
            display.Image(data=f.read(), format='png')




# velocity threshold is threshold for the tag_id to be considered resting
# resting_threshold is the min waiting time that can be considered as resting taken as 1 second here
def find_resting_time(data, tag_id, velocity_threshold, resting_threshold):
    # Preprocessing the data
    data = data[data['tag_id'] == tag_id].copy() # .copy is used to solve the 'SettingWithCopyWarning' warning
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data = data.sort_values(by='time', ascending=True)
    # creating the velocity column and storing the values between the current and the previous data
    data['velocity'] = ((data['x'].diff() ** 2 + data['y'].diff() ** 2) ** 0.5) / data['time'].diff().dt.total_seconds()
    resting_periods = []
    start_time = None
    current_date = None
    for i in range(len(data)):
        # Change of date marks the end of the resting period
        if data.iloc[i]['time'].date() != current_date:
            if start_time is not None:
                end_time = data.iloc[i-1]['time']
                resting_periods.append((start_time, end_time))
            current_date = None
            start_time = None
        # if velocity less than the threshold either wait to end or set start time
        if data.iloc[i]['velocity'] < velocity_threshold:
            if start_time is None:
                start_time = data.iloc[i]['time']
                current_date = data.iloc[i]['time'].date()
        else: # if velocity is not less than threshold then mark this time as end_time
            if start_time is not None:
                end_time = data.iloc[i-1]['time']
                if (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds() > resting_threshold:
                    resting_periods.append((start_time, end_time))
                start_time = None
    # finally if the last time is not ended then last data will be the end time
    if start_time is not None:
        end_time = data.iloc[-1]['time']
        resting_periods.append((start_time, end_time))
    return resting_periods

def trajectory_data_generation(data,  args, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    # Sort the dataset by time
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data = data.sort_values(by='time', ascending=True)

    # Creating empty datasets to store the train, test, and validation data
    train = pd.DataFrame(columns=['tag_id', 'time', 'x', 'y', 'description', 'trajectory_name', 'point_type'])
    test = pd.DataFrame(columns=['tag_id', 'time', 'x', 'y', 'description', 'trajectory_name', 'point_type'])
    val = pd.DataFrame(columns=['tag_id', 'time', 'x', 'y', 'description', 'trajectory_name', 'point_type'])
    types = ["start", "intermediate", "end"]

    # Iterating over each tag_id
    for tag_id in data['tag_id'].unique():
        print(f'Preprocessing for {tag_id}')
        tag_data = data[data['tag_id'] == tag_id]
        
        # Calculate the time differences
        # time_diffs = data['time'].diff().dropna()

        # Calculate the average of time differences
        # average_time_diff = time_diffs.mean()
        # t = 5 * average_time_diff
        t = pd.Timedelta(minutes=1, seconds = 15)
        # print(f"threshold = {t}")
        # Initialize variables
        starting_point = None
        trajectory_name = "trajectory_1"
        rows_to_add = []
        # Iterating over each data point for the current tag_id
        for i in tqdm(range(len(tag_data))):
            prev_point = (tag_data.iloc[i - 1]['x'], tag_data.iloc[i - 1]['y']) if i > 0 else None
            current_point = (tag_data.iloc[i]['x'], tag_data.iloc[i]['y'])
            time_diff_curr_prev = tag_data.iloc[i]['time'] - tag_data.iloc[i - 1]['time'] if i > 0 else None
            # Check if the time difference is less than t
            if time_diff_curr_prev is not None and time_diff_curr_prev < t:
                # Check if the current point is negative
                if (current_point[0] < 0 or current_point[1] < 0) and starting_point is not None:
                    # Assign the prev_point as the end point and increase trajectory_name
                    end_point = prev_point
                    rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i - 1]['time'], 'x': end_point[0],
                                        'y': end_point[1], 'description': tag_data.iloc[i]['description'],
                                        'trajectory_name': trajectory_name, 'point_type': types[2]})
                    if len(rows_to_add) > 40:
                        # Determine the target dataset based on probability ratios
                        rand_num = random.random()
                        if rand_num < train_ratio:
                            train = pd.concat([train, pd.DataFrame(rows_to_add)])
                        elif rand_num < (train_ratio + test_ratio):
                            test = pd.concat([test, pd.DataFrame(rows_to_add)])
                        else:
                            val = pd.concat([val, pd.DataFrame(rows_to_add)])
                        trajectory_name = "trajectory_" + str(int(trajectory_name.split("_")[1]) + 1)
                    rows_to_add = []
                    starting_point = None
                else:
                    # Add the current point to the rows_to_add list
                    if starting_point is not None:
                        rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i]['time'], 'x': current_point[0],
                                            'y': current_point[1], 'description': tag_data.iloc[i]['description'],
                                            'trajectory_name': trajectory_name, 'point_type': types[1]})

                    elif current_point[0] > 0 and current_point[1] > 0:
                        starting_point = current_point
                        rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i]['time'], 'x': current_point[0],
                                            'y': current_point[1], 'description': tag_data.iloc[i]['description'],
                                            'trajectory_name': trajectory_name, 'point_type': types[0]})

            else:
                # Check if rows_to_add has any data before changing the trajectory_name
                if len(rows_to_add) > 40:
                    rows_to_add[-1]['point_type'] = types[2]
                    # Determine the target dataset based on probability ratios
                    rand_num = random.random()
                    if rand_num < train_ratio:
                        train = pd.concat([train, pd.DataFrame(rows_to_add)])
                    elif rand_num < (train_ratio + test_ratio):
                        test = pd.concat([test, pd.DataFrame(rows_to_add)])
                    else:
                        val = pd.concat([val, pd.DataFrame(rows_to_add)])
                    trajectory_name = "trajectory_" + str(int(trajectory_name.split("_")[1]) + 1)
                rows_to_add = []
                
                # Make starting_point None and check existing conditions
                starting_point = None

                if current_point[0] > 0 and current_point[1] > 0:
                    starting_point = current_point
                    rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i]['time'], 'x': current_point[0],
                                        'y': current_point[1], 'description': tag_data.iloc[i]['description'],
                                        'trajectory_name': trajectory_name, 'point_type': types[0]})

    # Check if rows_to_add has any remaining data
    if len(rows_to_add) > 40:
        rows_to_add[-1]['point_type'] = types[2]
        # Determine the target dataset based on probability ratios
        rand_num = random.random()
        if rand_num < train_ratio:
            train = pd.concat([train, pd.DataFrame(rows_to_add)])
        elif rand_num < (train_ratio + test_ratio):
            test = pd.concat([test, pd.DataFrame(rows_to_add)])
        else:
            val = pd.concat([val, pd.DataFrame(rows_to_add)])

    # Write the datasets to respective files
    split = {'train':
                train,
            'test':
                test,
            'val':
                val}
    toNumpy = Data2Numpy(args.subset, args.past_length, args.future_length, split)
    toNumpy.generate_data()



def trajectory_data_generation_no_rest(data, args, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    # Sort the dataset by time
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data = data.sort_values(by='time', ascending=True)

    # Creating empty datasets to store the train, test, and validation data
    train = pd.DataFrame(columns=['tag_id', 'time', 'x', 'y', 'description', 'trajectory_name', 'point_type'])
    test = pd.DataFrame(columns=['tag_id', 'time', 'x', 'y', 'description', 'trajectory_name', 'point_type'])
    val = pd.DataFrame(columns=['tag_id', 'time', 'x', 'y', 'description', 'trajectory_name', 'point_type'])
    types = ["start", "intermediate", "end"]

    # Iterating over each tag_id
    for tag_id in data['tag_id'].unique():
        print(f'Preprocessing for {tag_id}')
        tag_data = data[data['tag_id'] == tag_id]

        # finding resting periods to iterate over and exclude the points
        resting_time = find_resting_time(data, tag_id, 0.5, 10)
        resting_periods = [(pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S'),
                            pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S')) for start, end in resting_time]
        
        t = pd.Timedelta(minutes=1, seconds = 15)

        # Initialize variables
        starting_point = None
        trajectory_name = "trajectory_1"
        rows_to_add = []
        start_loc = 0
        # Iterating over each data point for the current tag_id
        for i in tqdm(range(len(tag_data))):
            skip = False
            prev_point = (tag_data.iloc[i - 1]['x'], tag_data.iloc[i - 1]['y']) if i > 0 else None
            current_point = (tag_data.iloc[i]['x'], tag_data.iloc[i]['y'])
            for start, end in resting_periods[start_loc:]:
                # condition that current point lies in the interval
                if start <= pd.to_datetime(tag_data.iloc[i]['time']) <= end:
                    skip = True
                    break
                # condition that point lies between the previous and the current interval
                elif pd.to_datetime(tag_data.iloc[i]['time']) < start:
                    break
                start_loc += 1  # start_loc allows to optimize the above process by not doing redundant iterations

            if skip:
                # if point lies in the interval and if there is a continued trajectory then mark the end point as previous point
                if starting_point is not None:
                    # Assign the prev_point as the end point and increase trajectory_name
                    end_point = prev_point
                    rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i - 1]['time'], 'x': end_point[0],
                                        'y': end_point[1], 'description': tag_data.iloc[i]['description'],
                                        'trajectory_name': trajectory_name, 'point_type': types[2]})
                    if len(rows_to_add) > 40:
                        # Determine the target dataset based on probability ratios
                        rand_num = random.random()
                        if rand_num < train_ratio:
                            train = pd.concat([train, pd.DataFrame(rows_to_add)])
                        elif rand_num < (train_ratio + test_ratio):
                            test = pd.concat([test, pd.DataFrame(rows_to_add)])
                        else:
                            val = pd.concat([val, pd.DataFrame(rows_to_add)])
                        trajectory_name = "trajectory_" + str(int(trajectory_name.split("_")[1]) + 1)
                    rows_to_add = []
                    starting_point = None
            else:
                time_diff_curr_prev = tag_data.iloc[i]['time'] - tag_data.iloc[i - 1]['time'] if i > 0 else None
                # Check if the time difference is less than t
                if time_diff_curr_prev is not None and time_diff_curr_prev < t:
                    # Check if the current point is negative
                    if (current_point[0] < 0 or current_point[1] < 0) and starting_point is not None:
                        # Assign the prev_point as the end point and increase trajectory_name
                        end_point = prev_point
                        rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i - 1]['time'], 'x': end_point[0],
                                            'y': end_point[1], 'description': tag_data.iloc[i]['description'],
                                            'trajectory_name': trajectory_name, 'point_type': types[2]})
                        if len(rows_to_add) > 40:
                            # Determine the target dataset based on probability ratios
                            rand_num = random.random()
                            if rand_num < train_ratio:
                                train = pd.concat([train, pd.DataFrame(rows_to_add)])
                            elif rand_num < (train_ratio + test_ratio):
                                test = pd.concat([test, pd.DataFrame(rows_to_add)])
                            else:
                                val = pd.concat([val, pd.DataFrame(rows_to_add)])
                            trajectory_name = "trajectory_" + str(int(trajectory_name.split("_")[1]) + 1)
                        rows_to_add = []
                        starting_point = None
                    else:
                        # Add the current point to the rows_to_add list
                        if starting_point is not None:
                            rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i]['time'], 'x': current_point[0],
                                                'y': current_point[1], 'description': tag_data.iloc[i]['description'],
                                                'trajectory_name': trajectory_name, 'point_type': types[1]})

                        elif current_point[0] > 0 and current_point[1] > 0:
                            starting_point = current_point
                            rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i]['time'], 'x': current_point[0],
                                                'y': current_point[1], 'description': tag_data.iloc[i]['description'],
                                                'trajectory_name': trajectory_name, 'point_type': types[0]})

                else:
                    # Check if rows_to_add has any data before changing the trajectory_name
                    if len(rows_to_add) > 40:
                        # Determine the target dataset based on probability ratios
                        rand_num = random.random()
                        if rand_num < train_ratio:
                            train = pd.concat([train, pd.DataFrame(rows_to_add)])
                        elif rand_num < (train_ratio + test_ratio):
                            test = pd.concat([test, pd.DataFrame(rows_to_add)])
                        else:
                            val = pd.concat([val, pd.DataFrame(rows_to_add)])
                        trajectory_name = "trajectory_" + str(int(trajectory_name.split("_")[1]) + 1)
                    rows_to_add = []
                    
                    # Make starting_point None and check existing conditions
                    starting_point = None

                    if current_point[0] > 0 and current_point[1] > 0:
                        starting_point = current_point
                        rows_to_add.append({'tag_id': tag_id, 'time': tag_data.iloc[i]['time'], 'x': current_point[0],
                                            'y': current_point[1], 'description': tag_data.iloc[i]['description'],
                                            'trajectory_name': trajectory_name, 'point_type': types[0]})

    # Check if rows_to_add has any remaining data
    if len(rows_to_add) > 40:
        rows_to_add[-1]['point_type'] = types[2]
        # Determine the target dataset based on probability ratios
        rand_num = random.random()
        if rand_num < train_ratio:
            train = pd.concat([train, pd.DataFrame(rows_to_add)])
        elif rand_num < (train_ratio + test_ratio):
            test = pd.concat([test, pd.DataFrame(rows_to_add)])
        else:
            val = pd.concat([val, pd.DataFrame(rows_to_add)])

    # Write the datasets to respective files
    split = {'train':
                train,
            'test':
                test,
            'val':
                val}
    toNumpy = Data2Numpy(args.subset, args.past, args.future, split)
    toNumpy.generate_data()

