# Einbinden der Bibliotheken
import torch
import torch.nn as nn
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
from pygame.locals import K_ESCAPE
import pickle
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import numpy as np
import collections
import cv2
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import pygame
import sys 
import subprocess
import time

# Benutzerdefinierte Wrapper-Erwweiterung
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


# Skalierung und Umwandlung in Graustufen
class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """

    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


# Änderung Reihenfolge der Achsen für PyTorch
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


# Frame wird normalisiert
class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


# Puffer um Frames zwischenzuspeichern
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


# Transformationen des Eingangsenvironments
def make_env(env):
    env = MaxAndSkipEnv(env)  # nur jedes 'skip'-te Frame wird zurückgegeben
    env = ProcessFrame84(env)  # Skalierung auf 84x84 und Umwandlung in Grauwert
    env = ImageToPyTorch(env)  # Änderung der Achsreihenfolge
    env = BufferWrapper(env, 4)  # Letzte 4 aufeinanderfolgende Frames werden gespeichert
    env = ScaledFloatFrame(env)  # Normalisierung der Frame-Pixel auf zwischen 0 und 1
    return JoypadSpace(env, RIGHT_ONLY)  # Aktionen werden auf Rechts-Aktionen beschränkt

# Unterklasse von nn.Module -> Vorbild Go-Explore Architektur
class CustomSolver(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(CustomSolver, self).__init__()
        # Definition der Faltungsschichten
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # Definition der Fully-Connected-Schichten
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 800),
            nn.ReLU(),
            nn.Linear(800, n_actions)
        )

    # Berechnung der Größe des Ausgabevektors nach Faltungsschichten
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    # Berechnung des Vorwärtspfads
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
        
class DownsampleFrame:
    """
    Downsamples the frame to a lower resolution.
    """

    def __init__(self, downsample_factor):
        self.downsample_factor = downsample_factor

    def downsample(self, frame):
        return cv2.resize(frame, (frame.shape[1] // self.downsample_factor, frame.shape[0] // self.downsample_factor))

##############################################################################################################################################

if len(sys.argv) < 2:
    level = "SuperMarioBros-1-1-v0"
    model = "model1"
    automatic_mode = "False"
else:
    level = sys.argv[1]
    model = sys.argv[2]
    automatic_mode = sys.argv[3]

downsample_factor = 1

# Initialize the DownsampleFrame object
downsampler = DownsampleFrame(downsample_factor)

env = gym_super_mario_bros.make(level)
env = make_env(env)  # Wraps the environment so that frames are grayscale
observation_space = env.observation_space.shape
action_space = env.action_space.n

print(action_space)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = CustomSolver(observation_space, action_space).to(device)

net.load_state_dict(torch.load(f"{model}.pt", map_location=torch.device(device)))

# Mapping model to model_info
model_info = {"model1": "1000 iterations",
              "1000it": "1000 iterations",
              "ea1000it": "1000 iterations",
              "model2": "2000 iterations",
              "2000it": "2000 iterations",
              "ea2000it": "2000 iterations",
              "model3": "8000 iterations",
              "8000it": "8000 iterations",
              "ea8000it": "2000 iterations",
              "model4": "Expert model",
              "austrainiert": "Expert model",
              "eaaustrainiert": "Expert model"}

# Mapping level to level_info
level_info = {"SuperMarioBros-1-1-v0": "Level 1-1",
	      "SuperMarioBros-1-2-v0": "Level 1-2",
              "SuperMarioBros-1-3-v0": "Level 1-3"}

# Initialize Pygame
pygame.init()

# Set the dimensions of the pygame window
window_width = pygame.display.Info().current_w
window_height = pygame.display.Info().current_h
window = pygame.display.set_mode((window_width, window_height), pygame.FULLSCREEN)
clock = pygame.time.Clock()

small_window_width = 1100
small_window_height = 1100

# Font settings for displaying text
font = pygame.font.Font(None, 44)
text_color = (255, 255, 255)

state = env.reset()

state = torch.Tensor([state])

total_reward = 0

steps = 0

info = None

pos_not_changed_count = 0

action_iteration = 0

# Clear the screen with any color
window.fill((0, 0, 0))

while True:
    
    action = torch.argmax(net(state.to(device))).unsqueeze(0).unsqueeze(0).cpu()
    
    print(int(action[0]))
    
    if action[0] == 0:
        pressed_keys_text = {""}
    elif action[0] == 1:
        pressed_keys_text = {"right"}
    elif action[0] == 2:
        pressed_keys_text = {"right, A"}
    elif action[0] == 3:
        pressed_keys_text = {"right, B"}
    elif action[0] == 4:
        pressed_keys_text = {"right, A, B"}

    steps += 1
    
    info_old = info
    
    state_next, reward, terminal, info = env.step(int(action[0]))
    total_reward += reward
    
    state_next = torch.Tensor([state_next])
    reward = torch.tensor([reward]).unsqueeze(0)
    terminal = torch.tensor([int(terminal)]).unsqueeze(0)
    
    state = state_next
    
    screen = env.render(mode='rgb_array')
    # Downsample the Pygame surface
    screen = downsampler.downsample(screen)
    screen = pygame.surfarray.make_surface(screen)
    screen = pygame.transform.flip(screen, True, False)  # Flip verticallyd
    screen = pygame.transform.rotate(screen, 90)  # Rotate counterclockwise by 90 degrees
    screen = pygame.transform.scale(screen, (small_window_width, small_window_height))
    
    
    window.blit(screen, (window_width // 2 - small_window_width // 2, window_height // 2 - small_window_height // 2 + 10)) # Adjust the position as needed
    
    # Show pressed keys
    text_surface = font.render("Pressed keys: {}".format(pressed_keys_text), True, text_color)
    window.blit(text_surface, (window_width // 2 - small_window_width // 2 + 10, window_height // 2 - small_window_height // 2 + 20))
    # Show Level and Model
    text2_surface = font.render("Level: {}, Model: {}".format(level_info[level], model_info[model]), True, text_color)
    window.blit(text2_surface, (window_width  // 2 + small_window_width // 2 - 600, window_height // 2 - small_window_height // 2 + 20))
    pygame.display.flip()
    clock.tick(60)
    
    if action_iteration != 0:
        if info['x_pos'] == info_old['x_pos']:
            pos_not_changed_count += 1
        else: 
            pos_not_changed_count = 0
        
    if pos_not_changed_count == 70:
        break
        
    if terminal:
        break
        
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
        
    action_iteration += 1
    

print("Total Reward is {}".format(total_reward))

score_font = pygame.font.Font(None, 36)

# Display the score in the score window
score_text = font.render("Total Reward: {}".format(total_reward), True, text_color)
window.fill((0, 0, 0))
window.blit(score_text, (window_width // 2 - score_text.get_width() // 2,
                         window_height // 2 - score_text.get_height() // 2))
pygame.display.flip()

time.sleep(4)

if automatic_mode == "True":
    pygame.quit()
    subprocess.call(["python", "automatic_mode.py", level, model])
    sys.exit()

# Wait for a button press to close the score window
if automatic_mode != "True":    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pygame.quit()
                subprocess.call(["python", "GUI.py"])
                sys.exit()
            

