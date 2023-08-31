from testingDictSpace import BikeEnv
from gym.wrappers import FlattenObservation

env = BikeEnv()
env = FlattenObservation(env)

episodes = 3

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print("random action: ", random_action)
        obs, reward, done, info = env.step(random_action)
        print("reward: ", reward)