import metadrive  # Import this package to register the environment!
import gym
from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv(config=dict(use_render=True))
# env = gym.make("MetaDrive-10env-v0", config=dict(use_render=True))
env.reset()
for i in range(1000):
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        env.reset()
env.close()