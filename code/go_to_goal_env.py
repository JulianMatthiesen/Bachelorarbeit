#!/usr/bin/env python3

import glob
import math
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import gym
import numpy as np
from gym import spaces
import carla
import random


class BikeEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    
    XMAX=12.5
    XMIN=-9
    YMAX=-35
    YMIN=-124

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(BikeEnv, self).__init__()
        # Define action and observation space

        high = np.array([
            1.0,   # throttle bike
            1.0    # steer bike
        ])

        low = np.array([
            -1.0,   # throttle bike
            -1.0   # steer bike
        ])

        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        high = np.array([
            360,         # angle to target
            92        # distance 
        ])

        low = np.array([
            0,        # angle to target
            0         # distance
        ])

        obs_size = len(high)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)



        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        time.sleep(1)
        self.world.unload_map_layer(carla.MapLayer.All)
        self.bp_lib = self.world.get_blueprint_library()
        self.bike_bp = self.bp_lib.find("vehicle.diamondback.century")

        self.bike = self.spawn_bike()

        # set synchronous mode und fixed time-step 
        settings = self.world.get_settings()
        settings.synchronous_mode = True  
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.target_location = None
        self.done = False
        self.reward = 0
        self.ret = 0
        self.tick_count = 0
        self.max_time_steps = 1000
        self.world.tick()
        self.info = {"actions": []}

    def step(self, action):
        throttle = float((action[0] + 1)/ 2)
        steer=float(action[1])
        self.bike.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        # update bike_location
        self.bike_location = self.bike.get_transform().location
        
        self.world.debug.draw_string(self.target_location, "X", draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=0.1,
                                     persistent_lines=True)

        self.info["actions"].append(action.tolist())
        observation = self.get_observation()
        self.reward = self.calculate_reward(action)
        self.ret += self.reward
        return observation, self.reward, self.done, self.info

    def reset(self):
        if self.tick_count != 0:
            print("\nThis Episode: ")
            print("Steps: " + str(self.tick_count))
            print("Return: ", round(self.ret))

        if not len(self.world.get_actors()) == 0:
            self.bike.destroy()

        self.bike = self.spawn_bike()
        
        self.done = False
        self.reward = 0
        self.ret = 0
        self.tick_count = 0
        self.info = {"actions": []}
        self.world.tick()
        self.tick_count += 1
        return self.get_observation() 

    def close(self):
        self.bike.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
         
        self.world.tick()

    """
    def render(self):
        ...
    """

# ================== Helper Methods ==================

    def spawn_bike(self):        
        # spawn bike and set target location
        spawn_transform  = self.get_random_spawn_point()
        spawn_point_target = self.get_random_spawn_point()
        bike = self.world.try_spawn_actor(self.bike_bp, spawn_transform)
        self.target_location = spawn_point_target.location

        # if spawn failed
        while bike == None:
            spawn_transform  = self.get_random_spawn_point()
            spawn_point_target = self.get_random_spawn_point()
            bike = self.world.try_spawn_actor(self.bike_bp, spawn_transform)
            self.target_location = spawn_point_target.location

        self.bike_location = spawn_transform.location
        self.target_location = spawn_point_target.location

        return bike

    def get_random_spawn_point(self):
        # spawn vehicle at random location within square
        xSpawn = random.uniform(self.XMIN + 2, self.XMAX - 2)
        ySpawn = random.uniform(self.YMIN + 2, self.YMAX - 2)
        location = carla.Location(x=xSpawn, y=ySpawn, z=0.05)
        phiSpawn = random.uniform(-180, 180)
        rotation = carla.Rotation(pitch=0.0, yaw=phiSpawn, roll=0.0)
        random_point = carla.Transform(location, rotation)
        return random_point
    
    def get_observation(self):
        angle_to_target = [self.get_angle_to_target()]
        observated_dist = [self.get_distance_to_target()]
        observation = angle_to_target + observated_dist 
        observation = np.array(observation, dtype=np.float32)
        return observation

    def get_angle_to_target(self):
        # compute and return angle between forward vector and vector to target
        bike_transform = self.bike.get_transform()
        vector_to_target = self.target_location - bike_transform.location
        forward_vector  = bike_transform.get_forward_vector()
        angle_to_target = math.degrees(math.atan2(forward_vector.y, forward_vector.x) - math.atan2(vector_to_target.y, vector_to_target.x))
        angle_to_target = angle_to_target % 360
        return angle_to_target

    def get_distance_to_target(self):
        return self.bike_location.distance(self.target_location)
        
    def calculate_reward(self, action):
        # calculate Distance Reward
        current_distance = self.get_distance_to_target()
        distance_reward = -1.0 * (current_distance / 140)                       # normalized to [-1;0]
        
        # calculate Steering Reward
        steering = action[1] 
        angle_to_target = self.get_angle_to_target()                            # in [0°;360°]
                     
        if angle_to_target <= 180:                                              # 0.1° -> target minimal left; 179° -> target far left
            target_steering = max(-1.0 * (angle_to_target / 90.0), -1)          # big angles -> steer should be close to -1; limit at -1
            steering_difference = abs(target_steering - steering)               # in [0;2]
            steering_reward = (1 - steering_difference) * 0.25                  # in[-0.25;0.25]

        if angle_to_target > 180:                                               # 359° -> target minimal right; 181° -> target far right
            target_steering = min(1.0 * ((360 - angle_to_target ) / 90.0), 1)   # small angles (close to 180°) -> steer should be close to 1; limit at 1
            steering_difference = abs(target_steering - steering)               # in [0;2]
            steering_reward = (1 - steering_difference) * 0.25                  # in[-0.25;0.25]
        
        reward = (distance_reward + steering_reward)
   
        # end episode, when reaching the target, reaching time limit or leaving the square 
        if current_distance < 1.0:
            print("\ntarget reached!")
            self.done = True
            reward = 100
        
        elif self.tick_count >= self.max_time_steps:
            print("\nmax_time_steps reached!")
            self.done = True
            reward = -100

        elif not self.is_within_bounds():
             print("\nout of bounds!")
             self.done = True
             reward = 0

        self.world.tick()
        self.tick_count += 1

        return reward
    
    def is_within_bounds(self):
        return self.XMIN <= self.bike_location.x <= self.XMAX and self.YMIN <= self.bike_location.y <= self.YMAX

    
