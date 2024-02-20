import numpy as np
import math
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar
from matplotlib.patches import Rectangle
import json

def write_data_to_file(data, filename):
    # Convert all tuples in the data to lists for JSON compatibility
    json_compatible_data = [[list(item) for item in sublist] for sublist in data]

    # Write the data to the specified file
    with open(filename, 'w') as file:
        json.dump(json_compatible_data, file)

def cartesian_to_polar(velocity_x, velocity_y):
    # Calculate the magnitude of the velocity
    magnitude = math.sqrt(velocity_x**2 + velocity_y**2)
    
    # Calculate the angle in radians and convert to degrees
    # atan2 is used instead of atan to handle cases where velocity_x is 0
    angle = math.atan2(velocity_y, velocity_x) * (180 / np.pi)
    
    return round(magnitude, 2), round(angle, 2)

# ====================== PATH GENERATION ALGORITHM ==============================================================================

def interpolate_path(points, delta_time, velmax, accmax):
    # Extract x and y coordinates from the given points
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    # Perform cubic spline interpolation with initial and ending velocity constraints
    t = np.linspace(0, 1, len(points))
    cs_x = CubicSpline(t, x, bc_type='clamped')
    cs_y = CubicSpline(t, y, bc_type='clamped')

    # Derive cs_x and cs_y to obtain velocity functions
    vx = cs_x.derivative()
    vy = cs_y.derivative()

    # Derive vy and vy to obtain acceleration functions
    ax = vx.derivative()
    ay = vy.derivative()

    # Function to compute the magnitude of the velocity vector
    v = lambda t: np.sqrt(vx(t)**2 + vy(t)**2)

    # Function to compute the magnitude of the velocity vector
    a = lambda t: np.sqrt(ax(t)**2 + ay(t)**2)

    # Define the objective function for minimization (negative of velocity magnitude)
    objective_function_v = lambda t: -v(t)
    objective_function_a = lambda t: -v(t)

    # Set the time range over which to search for the maximum
    time_range = (0, 1)  # Adjust the range based on your specific time domain

    # Find the time at which the velocity magnitude is maximized
    result_v = minimize_scalar(objective_function_v, bounds=time_range, method='bounded')
    result_a = minimize_scalar(objective_function_a, bounds=time_range, method='bounded')

    # The result.x gives the time at which the maximum velocity occurs
    max_velocity_time = result_v.x
    max_acceleration_time = result_a.x
    max_velocity = -result_v.fun  # Note: Negate to get the actual maximum velocity magnitude
    max_acceleration = -result_a.fun  # Note: Negate to get the actual maximum acceleration magnitude

    adjusted_time = max(max_velocity/velmax, np.sqrt(1000*max_acceleration/accmax))
    number_of_iterations = int(adjusted_time/delta_time)
    total_time = number_of_iterations * delta_time

    # Generate interpolated points with the specified time step
    t_interpolated = np.arange(0, 1, 1/number_of_iterations)
    x_interpolated = cs_x(t_interpolated)
    y_interpolated = cs_y(t_interpolated)
    vx_interpolated = vx(t_interpolated)/adjusted_time
    vy_interpolated = vy(t_interpolated)/adjusted_time

    # Combine x and y coordinates of interpolated points
    interpolated_path = list(zip(x_interpolated, y_interpolated))
    velocity = list(zip(vx_interpolated, vy_interpolated))
    polar_velocity = []

    ang = 0
    for vx, vy in zip(vx_interpolated, vy_interpolated):
        v_mag, v_ang = cartesian_to_polar(vx, vy)
        ang += v_ang * delta_time / 1000
        polar_velocity.append((v_mag, v_ang, round(ang, 2)))

    return interpolated_path, velocity, polar_velocity, total_time

def multiple_interpolated_paths_and_times(trajectories, velmax, accmax, delta_time):
    paths = []
    total_time_of_whole_path = 0
    velocities = []
    polar_velocities = []
    i = 0

    for trajectory in trajectories:
        path, velocity, polar_velocity, total_time = interpolate_path(trajectory, delta_time, velmax, accmax)
        total_time_of_whole_path += total_time
        paths.append(path)
        velocities.append(velocity)
        polar_velocities.append(polar_velocity)
        i+=1
        print("number of points in trajectory ",i," : ", len(path))

    print("Time to travel: ", total_time_of_whole_path, " ms")

    return paths, velocities, polar_velocities, total_time_of_whole_path

# ====================== ENDING OF PATH GENERATION ALGORITHM ==========================================================================
    
# Example usage:
# trajectory1 = [(0, 0), (1, 3), (2, 1), (0, 2), (4, 1)]  # Replace with your set of points
# trajectory2 = [(10, 0), (0, 10)]
delta_time = 10
velmax = 2 # meter per second ==================================================================
accmax = 1 # meter per second square ============================================================

#======================================== AREA 1 POINT MAP INITIATION =====================================

start_R1_s1 = (1250, 350)
end_R1_s1 = (500, 3650)

wheat_radius = 45/2

R1_dim = (900, 900)

distance_from_center_to_grabing = 400

wheat_takes = [(2250, 125), (2500, 125), (2750, 125), (3000, 125), (3250, 125), (3500, 125),
                (3750, 125), (4000, 125), (4250, 125), (4500, 125), (4750, 125), (5000, 125)]


wheat_grabs = [((wheat_takes[0][0] + wheat_takes[1][0] + wheat_takes[2][0] + wheat_takes[3][0])/4, wheat_takes[0][1] + distance_from_center_to_grabing), 
               ((wheat_takes[4][0] + wheat_takes[5][0] + wheat_takes[6][0] + wheat_takes[7][0])/4, wheat_takes[4][1] + distance_from_center_to_grabing), 
               ((wheat_takes[8][0] + wheat_takes[9][0] + wheat_takes[10][0] + wheat_takes[11][0])/4, wheat_takes[8][1] + distance_from_center_to_grabing)]

wheat_plant = [(2345, 2250), (2845, 2250), (3345, 2250), (3845, 2250), (4345, 2250), (4845, 2250),
                (2345, 2750), (2845, 2750), (3345, 2750), (3845, 2750), (4345, 2750), (4845, 2750)]

wheat_drop = [((wheat_plant[0][0] + wheat_plant[1][0])/2, wheat_plant[0][1] - distance_from_center_to_grabing),
              ((wheat_plant[2][0] + wheat_plant[3][0])/2, wheat_plant[2][1] - distance_from_center_to_grabing),
              ((wheat_plant[4][0] + wheat_plant[5][0])/2, wheat_plant[4][1] - distance_from_center_to_grabing),
              ((wheat_plant[6][0] + wheat_plant[7][0])/2, wheat_plant[6][1] - distance_from_center_to_grabing),
              ((wheat_plant[8][0] + wheat_plant[9][0])/2, wheat_plant[8][1] - distance_from_center_to_grabing),
              ((wheat_plant[10][0] + wheat_plant[11][0])/2, wheat_plant[10][1] - distance_from_center_to_grabing)]

take_4_wheats_2 = [start_R1_s1,(start_R1_s1[0]*0.5 + wheat_grabs[2][0]*0.5, start_R1_s1[1]*0.5 + wheat_grabs[2][1]*0.5 + 300),wheat_grabs[2]]
plant_5 = [wheat_grabs[2], wheat_drop[5]]
plant_2 = [wheat_drop[5], wheat_drop[2]]
take_4_wheats_1 = [wheat_drop[2],wheat_grabs[1]]
plant_4 = [wheat_grabs[1], wheat_drop[4]]
plant_1 = [wheat_drop[4], wheat_drop[1]]
take_4_wheats_0 = [wheat_drop[1],wheat_grabs[0]]
plant_3 = [wheat_grabs[0], wheat_drop[3]]
plant_0 = [wheat_drop[3], wheat_drop[0]]
end_stage_1 = [wheat_drop[0], (wheat_drop[0][0]*0.5 + end_R1_s1[0]*0.5, wheat_drop[0][1]*1.1 + end_R1_s1[1]*(-0.1)), end_R1_s1]

trajectories_area1 = [take_4_wheats_2, plant_5, plant_2, take_4_wheats_1, plant_4, plant_1, take_4_wheats_0, plant_3, plant_0, end_stage_1]

#======================================== END OF AREA 1 POINT MAP INITIATION =====================================

#======================================== AREA 2 POINT MAP INITIATION =====================================

start_R2 = end_R1_s1

distance_from_center_to_take_ball = 400

real_ball_y = [5190, 5690]
real_ball_x = [2375, 2875, 3375, 3875, 4375, 4875]

real_balls = []
for y in real_ball_y:
    for x in real_ball_x:
        real_balls.append((x, y))


ball_y = [5190 - distance_from_center_to_take_ball, 5690 + distance_from_center_to_take_ball]
ball_x = [2375, 2875, 3375, 3875, 4375, 4875]

balls = []
for y in ball_y:
    for x in ball_x:
        balls.append((x, y))

good_ball_index = [0, 1, 2, 6, 7, 8]

bad_ball_index = []
for index in range(12):
    if not index in good_ball_index:
        bad_ball_index.append(index)

purple_balls = []
for index in bad_ball_index:
    purple_balls.append(real_balls[index])
red_balls = []
for index in good_ball_index:
    red_balls.append(real_balls[index])
position_throw = (1000, 7500)

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

min = 10000000000
first_i = -1
for i in range(6):
    d = distance(start_R2, balls[bad_ball_index[i]])
    if min > d:
        min = d
        first_i = i

current_ball_index = first_i

taken = []
trajectories_area2 = [[start_R2, balls[current_ball_index]]]

type = False
for turn in range(12):
    #bad ball
    if current_ball_index < 6:
        ball_to_A = [balls[current_ball_index], (position_throw[0] + 1000, balls[current_ball_index][1] - 500), position_throw]
    else:
        ball_to_A = [balls[current_ball_index], position_throw]
    trajectories_area2.append(ball_to_A)
    taken.append(current_ball_index)

    if turn == 11:
        break

    #good ball
    min = 100000000000000
    next_ball_index = -1
    if type == False:
        check = bad_ball_index
        type = True
    else:
        check = good_ball_index
        type = False
    for t in check:
        if not t in taken:
            d = distance(position_throw, balls[t])
            if min > d:
                min = d
                next_ball_index = t

    current_ball_index = next_ball_index
    if current_ball_index < 6:
        A_to_ball = [position_throw,(position_throw[0] + 1000, balls[current_ball_index][1] - 500), balls[current_ball_index]]
    else:
        A_to_ball = [position_throw, balls[current_ball_index]]
    
    trajectories_area2.append(A_to_ball)


#======================================== END OF AREA 2 POINT MAP INITIATION =====================================
    
def read_data_from_file(self, filename):
    # Read the data from the specified file
    with open(filename, 'r') as file:
        json_compatible_data = json.load(file)

    # Convert lists back to tuples
    data = [[tuple(item) for item in sublist] for sublist in json_compatible_data]
    return data


trajectories = trajectories_area1

all_trajectory = []
for aa in trajectories:
    all_trajectory += aa

paths, velocities, polar_velocity, total_times = multiple_interpolated_paths_and_times(trajectories, velmax, accmax, delta_time)

# Filename to write and read the data
filename = 'velocity_data.txt'

# Load data from a file when the node is initialized
filename = '/home/willoweyes/BK_Robotics/ros2_ws/src/velocity_data.txt'  # Specify the path to your JSON file
central_vel_data = read_data_from_file(filename)


write_data_to_file(polar_velocity, filename)