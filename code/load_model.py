import math
import numpy as np
from stable_baselines3 import PPO   
from collision_avoidance_env import BikeEnv
import carla



env = BikeEnv()
env.reset()

# determine model to evaluate
models_dir = "models/modelName"
model_path = f"{models_dir}/1970000.zip"
model = PPO.load(model_path, env=env)

# set spectator view 
spectator = env.world.get_spectator()
spectator.set_transform(carla.Transform(carla.Location(x=-20.631437, y=-98.168648, z=90.807022), carla.Rotation(pitch=-74.297012, yaw=178.924805, roll=0.000012)))

# initialize evaluation parameters
average_speeds = []
time_steps_needed = []
target_reached_count = 0
collision_count = 0
max_time_steps_count = 0
out_of_bounds_count = 0
episodes = 100


# run model 
for ep in range(episodes):
    obs = env.reset()
    done = False
    episode_speeds = []
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        v = env.bike.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        episode_speeds.append(kmh)  # append speed in every step
        
    # check reason for end of an episode
    if env.done:
        if env.reward == 300:
            target_reached_count += 1
        elif env.reward == -100:
            collision_count += 1
        elif env.reward == -99:     # environment modified for evaluation
            max_time_steps_count += 1
        elif env.reward == 0:
            out_of_bounds_count += 1

    # average_speed of this episode
    if episode_speeds:
        average_speeds.append(sum(episode_speeds) /len(episode_speeds))

    # time steps of this episode
    time_steps_needed.append(env.tick_count)

print(f"Episoden mit Erfolg: {target_reached_count}")
print(f"Episoden mit Kollision: {collision_count}")
print(f"Episoden mit max_time_steps erreicht: {max_time_steps_count}")
print(f"Episoden mit out of bounds: {out_of_bounds_count}")
print(f"Durchschnittliche Geschwindigkeit über alle Episoden: {sum(average_speeds)/len(average_speeds)} km/h")
print(f"Durchschnittliche erreichte time steps über alle Episoden: {sum(time_steps_needed)/len(time_steps_needed)}")

# optional: print policy of the model
print(model.policy)

# optional: print number of weights 
model_parameters = filter(lambda p: p.requires_grad, model.policy.parameters())
weight_count = sum(p.numel() for p in model_parameters)
print(f"Anzahl der Gewichte im Modell: {weight_count}")







