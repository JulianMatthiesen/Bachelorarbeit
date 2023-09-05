#!/usr/bin/env python3

import datetime
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
    
    XMAX=13.5
    XMIN=-10
    YMAX=-35
    YMIN=-124
    DISCOUNT = 0.99
    

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(BikeEnv, self).__init__()
        # Define action and observation space

        high = np.array([
            1.0,   # throttle/brake bike
            1.0    # steer bike
        ])

        low = np.array([
            -1.0,   # throttle/brake bike
            -1.0   # steer bike
        ])

        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        high = np.array([
            1,         # direction to target
            92,        # distance 
            1,         # rotation bike 
            self.XMAX, self.YMAX    # position bike(x,y)
        ])

        low = np.array([
            -1,        # direction to target
            0,         # distance
            -1,        # rotation bike 
            self.XMIN, self.YMIN  # position bike(x,y)

             
        ])
        obs_size = len(high)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)


        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        time.sleep(1)
        self.world.unload_map_layer(carla.MapLayer.All)
        blueprint_library = self.world.get_blueprint_library()
        
        self.bike_bp = blueprint_library.find("vehicle.diamondback.century")
        self.target_pylon_bp = blueprint_library.find('static.prop.gnome')
        self.target_pylon = None

        # synchronous mode und Fixed time-step später wichtig für synchrone Sensoren
        settings = self.world.get_settings()
        settings.synchronous_mode = True  
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        #self.client.reload_world(False)

        #position spectator
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=23.243340, y=-113.735214, z=22.266951), carla.Rotation(pitch=-34.642471, yaw=147.506561, roll=0.000042)))
        spawn_point = self.get_random_spawn_point()
        self.bike = self.world.spawn_actor(self.bike_bp, spawn_point)
        self.bike_location = spawn_point.location

        self.info = {"actions": [],
                     "target_locations": []}
        self.target_location = self.set_new_target()
        self.done = False
        self.reward = 0
        self.ret = 0
        self.tick_count = 0
        self.max_time_steps = 3000
        self.world.tick()


    def step(self, action):
        if action[0] > 0:
            throttle = float(action[0]) 
            brake = 0.0
        else: 
            brake = float(action[0]) * (-1)
            throttle = 0.0

        steer=float(action[1])
        
        self.bike.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

        # update bike_location
        self.bike_location = self.bike.get_transform().location
        self.info["actions"].append(action.tolist())
        observation = self.get_observation()
        self.reward = self.calculate_reward(action)
        return observation, self.reward, self.done, self.info

    def reset(self):
        if not len(self.world.get_actors()) == 0:
            self.bike.destroy()
        spawn_point = self.get_random_spawn_point()
        self.bike = self.world.spawn_actor(self.bike_bp, spawn_point)
        self.bike_location = spawn_point.location
        self.world.tick()
        self.info = {"actions": [],
                     "target_locations": []}

        # set target at random location within square
        self.target_location = self.set_new_target()
    
        self.prev_distance = self.get_distance_to_target()
        self.done = False
        self.reward = 0
        self.ret = 0
        print("tick_count: " + str(self.tick_count))
        self.tick_count = 0
        
        self.world.tick()
        self.tick_count += 1
        return self.get_observation() #info

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
# ================== Hilfsmethoden ==================

    
    def get_random_spawn_point(self):
        # spawn vehicle at random location within square
        xSpawn = random.uniform(self.XMIN, self.XMAX)
        ySpawn = random.uniform(self.YMIN, self.YMAX-85)
        location = carla.Location(x=xSpawn, y=ySpawn, z=0.05)
        phiSpawn = random.uniform(55, 125)
        rotation = carla.Rotation(pitch=0.0, yaw=phiSpawn, roll=0.0)
        random_point = carla.Transform(location, rotation)
        return random_point
    
   
    
    def get_observation(self):
        bike_transform = self.bike.get_transform()
        get_current_location = bike_transform.location
        current_location = [np.clip(get_current_location.x, self.XMIN, self.XMAX), np.clip(get_current_location.y, self.YMIN, self.YMAX)]
        target_location = [self.target_location.x, self.target_location.y]
        dx = target_location[0] - current_location[0]
        dy = target_location[1] - current_location[1]
        observated_direction = [np.arctan2(dy, dx) / np.pi]
        
        observated_dist = [self.get_distance_to_target()]
        observated_rotation = [(bike_transform.rotation.yaw / 180)] 

       

        observation = observated_direction + observated_dist + observated_rotation + current_location
        observation = np.array(observation, dtype=np.float32)
        return observation

    def get_distance_to_target(self):
        return self.bike_location.distance(self.target_location)
    
    
    def set_new_target(self):
        # set new target at random location within square

        # target wird vor dem Fahrrad gesetzt
        if self.target_pylon:
            self.target_pylon.destroy()    
        target_location = self.bike.get_transform().location + (self.bike.get_transform().get_forward_vector() * 15)        
        target_location.x = np.clip((target_location.x + random.uniform(-3, 3)), self.XMIN + 2, self.XMAX - 2) 
        target_location.y =  np.clip((target_location.y + random.uniform(-3, 3)), self.YMIN + 2, self.YMAX - 2) 

        self.target_pylon = self.world.spawn_actor(self.target_pylon_bp, carla.Transform(target_location))
        return target_location

    
    def calculate_reward(self, action):
        current_distance = self.get_distance_to_target()
        distance_reward = -(current_distance / 1000)

        throttle_reward = 0
        reward_for_target = 0

        if action[0] > 0:
            throttle_reward = 0.05

        # if target reached -> reward for finding and calculate new target 
        if current_distance < 5.0:
            self.target_location = self.set_new_target()
            reward_for_target = 100
            self.tick_count = 0
            print("target reached")

        reward = (reward_for_target  + distance_reward + throttle_reward) 
        
        # negative reward and stop episode, when leaving the square or reaching time limit
        self.world.tick()
        self.tick_count += 1
        if not self.is_within_boundary() or self.tick_count >= self.max_time_steps:
            self.done = True
            reward = -100

        #if self.tick_count % 10 == 0:
           # print("\nkmh: " + str(kmh))
            #print("\nreward: " + str(reward))
            #print("distance to target: " + str(current_distance))
        return reward
    
    def is_within_boundary(self):
        return self.XMIN <= self.bike_location.x <= self.XMAX and self.YMIN <= self.bike_location.y <= self.YMAX

    