import numpy as np
import pickle
import gym
import time
import matplotlib.pyplot as plt
from matplotlib import animation

print("hello")
file = open('/Users/kushagrakhatwani/Reinforcement_Learning/lab_assign/Lab_4/policy_montecarlo','rb')
policy = pickle.load(file)
file.close()

# print(policy)
def get_state(observation):
    pos_space = np.linspace(-1.2,0.6,20)
    vel_space = np.linspace(-.07,.07,20)
    pos,vel = observation
    pos_bin = np.digitize(pos,pos_space)
    vel_bin = np.digitize(vel,vel_space)

    return pos_bin,vel_bin

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


env = gym.make("MountainCar-v0")
observation = env.reset()
frames = []
for _ in range(1):
    done = False
    while not done:
        # env.render()
        frames.append(env.render(mode="rgb_array"))
        time.sleep(0.1)
        observation = get_state(observation)
        action = policy[(observation)]
        observation, reward, done, info = env.step(int(action))
    if done:
        observation = env.reset()
env.close()
save_frames_as_gif(frames)
