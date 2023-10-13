from collision_avoidance_env import BikeEnv
from gym.wrappers import FlattenObservation

"""check environment a second time by sampling actions"""

env = BikeEnv()
env = FlattenObservation(env)

episodes = 10

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print("random Action: ", random_action)
        obs, reward, done, info = env.step(random_action)
        print("observation: ", obs)
        
