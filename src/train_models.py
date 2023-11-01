# Einbinden der Bibliotheken
import torch
import torch.nn as nn
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from pygame.locals import K_ESCAPE
from tqdm import tqdm
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

# from pygame import display

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


# Unterklasse von nn.Module -> DQN-Modell
class DQNSolver(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        # Definition der Faltungsschichten
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # Definition der Fully-Connected-Schichten
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    # Berechnung der Größe des Ausgabevektors nach Faltungsschichten
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    # Berechnung des Vorwärtspfads, Q-Werte für jede Aktion
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


# Implementierung eines Agenten unter Verwendung des DQN-Modells
# Verwendung des DQN-Modells für Training und Entscheidungsfindung
# Verwaltung eines Erfahrungsspeichers
class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, double_dq, pretrained):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.double_dq = double_dq
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.double_dq:
            self.local_net = DQNSolver(state_space, action_space).to(self.device)
            self.target_net = DQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.local_net.load_state_dict(torch.load("dq1.pt", map_location=torch.device(self.device)))
                self.target_net.load_state_dict(torch.load("dq2.pt", map_location=torch.device(self.device)))

            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
            self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
            self.step = 0
        else:
            self.dqn = DQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.dqn.load_state_dict(torch.load("dq.pt", map_location=torch.device(self.device)))
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load("STATE_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE2_MEM.pt")
            self.DONE_MEM = torch.load("DONE_MEM.pt")
            with open("ending_position.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open("num_in_queue.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    # Erfahrungsspeicher wird ein Erfahrungstupel hinzugefügt
    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    # Zufällige Stichprobe aus Erfahrungswerten wird zurückgegeben
    def recall(self):
        # Randomly sample 'batch size' experiences
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]

        return STATE, ACTION, REWARD, STATE2, DONE

    # Aktion basierend auf aktuellem State wird ausgewählt, mit Wahrscheinlichkeit von Exploration-Rate wird eine zufällige Aktion gewählt
    # Ansonsten Aktion mit höchstem Q-Wert
    def act(self, state):
        # Epsilon-greedy action

        if self.double_dq:
            self.step += 1
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        if self.double_dq:
            # Local net is used for the policy
            return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    # Gewichte des lokalen Netzwerks werden in Zielnetzwerk kopiert
    def copy_model(self):
        # Copy local net weights into target net

        self.target_net.load_state_dict(self.local_net.state_dict())

    # Erfahrungswiederholungsverfahren??
    # Explorations-Rate wird nach jedem Erfahrungswiederholungsprozess abhängig von exploration_decay reduziert
    def experience_replay(self):

        if self.double_dq and self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        if self.double_dq:
            # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
            target = REWARD + torch.mul((self.gamma *
                                         self.target_net(STATE2).max(1).values.unsqueeze(1)),
                                        1 - DONE)

            current = self.local_net(STATE).gather(1, ACTION.long())  # Local net approximation of Q-value
        else:
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
            target = REWARD + torch.mul((self.gamma *
                                         self.dqn(STATE2).max(1).values.unsqueeze(1)),
                                        1 - DONE)

            current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)


# One-Hot-Kodierung der Aktion, eine Liste von Nullen außer an der Stelle der Aktion = 1
def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action

    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]


# Aktueller State des Env wird angezeigt
def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d %s" % (ep, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())


##################################################################Einpflegen eines Eigenen Modells sowie Agenten##################################################################

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


# Implementierung eines Agenten unter Verwendung des Custom-Modells
# Verwendung des Custom-Modells für Training und Entscheidungsfindung
# Verwaltung eines Erfahrungsspeichers


###################################################Class Neuroagent##############################################################################################################################

class NeuroAgent:

    # Definition der N Agenten und Environments
    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr, double_dq,
                 dropout, exploration_max, exploration_min, exploration_decay, pretrained, pt_number, ea):

        self.state_space = state_space
        self.action_space = action_space
        self.double_dq = double_dq
        self.pretrained = pretrained
        self.pt_number = pt_number
        self.ea = ea  # Flag for Evolutionary Algorithm
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not self.ea:
            print(ea)
            self.local_net = CustomSolver(state_space, action_space).to(self.device)
            self.target_net = CustomSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.local_net.load_state_dict(torch.load("rl1.pt", map_location=torch.device(self.device)))
                self.target_net.load_state_dict(torch.load("rl2.pt", map_location=torch.device(self.device)))

            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
            self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
            self.step = 0

        else:
            self.dqn = CustomSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.dqn.load_state_dict(
                    torch.load("ea{}.pt".format(pt_number), map_location=torch.device(self.device)))
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory only if rl-Agent
        if not self.ea:
            self.max_memory_size = max_memory_size
            if self.pretrained:
                self.STATE_MEM = torch.load("STATE_MEM.pt")
                self.ACTION_MEM = torch.load("ACTION_MEM.pt")
                self.REWARD_MEM = torch.load("REWARD_MEM.pt")
                self.STATE2_MEM = torch.load("STATE2_MEM.pt")
                self.DONE_MEM = torch.load("DONE_MEM.pt")
                with open("ending_position.pkl", 'rb') as f:
                    self.ending_position = pickle.load(f)
                with open("num_in_queue.pkl", 'rb') as f:
                    self.num_in_queue = pickle.load(f)
            else:
                self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
                self.ACTION_MEM = torch.zeros(max_memory_size, 1)
                self.REWARD_MEM = torch.zeros(max_memory_size, 1)
                self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
                self.DONE_MEM = torch.zeros(max_memory_size, 1)
                self.ending_position = 0
                self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    # Erfahrungsspeicher wird ein Erfahrungstupel hinzugefügt
    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    # Zufällige Stichprobe aus Erfahrungswerten wird zurückgegeben
    def recall(self):
        # Randomly sample 'batch size' experiences
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]

        return STATE, ACTION, REWARD, STATE2, DONE

    # Aktion basierend auf aktuellem State wird ausgewählt, mit Wahrscheinlichkeit von Exploration-Rate wird eine zufällige Aktion gewählt
    # Ansonsten Aktion mit höchstem Q-Wert
    def act(self, state):
        # Epsilon-greedy action
        if not self.ea:
            self.step += 1
        if not self.ea:
            if random.random() < self.exploration_rate:
                return torch.tensor([[random.randrange(self.action_space)]])
        if not self.ea:
            # Local net is used for the policy
            return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    # Erfahrungswiederholungsverfahren??
    # Explorations-Rate wird nach jedem Erfahrungswiederholungsprozess abhängig von exploration_decay reduziert

    def copy_model(self):
        # Copy local net weights into target net

        self.target_net.load_state_dict(self.local_net.state_dict())

    def experience_replay(self):

        if self.double_dq and self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()

        if self.double_dq:
            # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
            target = REWARD + torch.mul((self.gamma *
                                         self.target_net(STATE2).max(1).values.unsqueeze(1)),
                                        1 - DONE)

            current = self.local_net(STATE).gather(1, ACTION.long())  # Local net approximation of Q-value
        else:
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
            target = REWARD + torch.mul((self.gamma *
                                         self.dqn(STATE2).max(1).values.unsqueeze(1)),
                                        1 - DONE)

            current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

    # Initialize the weights randomly
    def weights_init(m, m2):
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)) and (
                isinstance(m2, nn.Conv2d) or isinstance(m2, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_uniform_(m2.weight)
            nn.init.zeros_(m.bias)
            nn.init.zeros_(m2.bias)


########################################Klasse NeuroAgentManager####################################################

class NeuroAgentManager:

    def __init__(self, it, pretrained, n_agents, training_mode, n_gutes_erbgut, epsilon, level):

        if n_gutes_erbgut > n_agents:
            print("Error: n_gutes_erbgut cant be bigger then n_agents")
            return

        self.it = it
        self.pretrained = pretrained
        self.n_agents = n_agents
        self.training_mode = training_mode
        self.n_gutes_erbgut = n_gutes_erbgut
        self.epsilon = epsilon
        self.level = level

        self.env = gym_super_mario_bros.make(self.level)
        self.env = make_env(self.env)  # Wraps the environment so that frames are grayscale
        observation_space = self.env.observation_space.shape
        action_space = self.env.action_space.n

        self.agent = [None] * n_agents

        for i in range(n_agents):
            if i == 0:
                ea = False
            else:
                ea = True

            self.agent[i] = NeuroAgent(state_space=observation_space,
                                       action_space=action_space,
                                       max_memory_size=30000,
                                       batch_size=32,
                                       gamma=0.9,
                                       lr=0.00025,
                                       double_dq=True,
                                       dropout=0.,
                                       exploration_max=1,
                                       exploration_min=0.02,
                                       exploration_decay=0.99,
                                       # double_dq=False,
                                       pretrained=pretrained,
                                       pt_number=i,
                                       ea=ea)

            # Wenn Pretrained-Flag gesetzt ist werden Gewichte für NN aus pt-File entnommen, siehe NeuroAgent.__init__()
            if ea == True and pretrained == False:
                self.agent[i].dqn.apply(self.agent[i].weights_init)

    # Sequentielles Starten der Agenten
    def start_agents(self):
        
        # Set the dimensions of the pygame window
        window_width = 800
        window_height = 600
        window = pygame.display.set_mode((window_width, window_height))
        clock = pygame.time.Clock()

        self.env.reset()
        log_reward = []
        last_rewards = [0] * self.n_agents  # List to store the last recorded reward for each agent

        for ep_num in tqdm(range(self.it)):

            total_rewards = []

            for ag_num in range(self.n_agents):
                state = self.env.reset()
                # state = np.array(state)
                state = torch.Tensor([state])
                total_reward = 0
                steps = 0

                while True:
                    action = self.agent[ag_num].act(state)
                    steps += 1

                    state_next, reward, terminal, info = self.env.step(int(action[0]))
                    total_reward += reward
                    state_next = torch.Tensor([state_next])
                    reward = torch.tensor([reward]).unsqueeze(0)
                    terminal = torch.tensor([int(terminal)]).unsqueeze(0)

                    if ag_num == 0:
                        self.agent[0].remember(state, action, reward, state_next, terminal)
                        self.agent[0].experience_replay()

                    state = state_next

                    screen = self.env.render(mode='rgb_array')
                    screen = pygame.surfarray.make_surface(screen)
                    screen = pygame.transform.flip(screen, True, False)  # Flip verticallyd
                    screen = pygame.transform.rotate(screen, 90)  # Rotate counterclockwise by 90 degrees
                    screen = pygame.transform.scale(screen, (window_width, window_height))

                    window.blit(screen, (0, 0))
                    pygame.display.flip()
                    clock.tick(20)

                    # print(info['x_pos'], info['time'])
                    if ((info['x_pos'] < 50) and info['time'] < 280):
                        break

                    if terminal:
                        break
                    
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == K_ESCAPE:
                                pygame.quit()

                    if ep_num != 0:
                        if ag_num in best_agents_ind:  # Checking if the agent is part of the best agents
                            break
                
                if ep_num != 0:
                    if ag_num in best_agents_ind:  # Checking if the agent is part of the best agents
                        total_reward = last_rewards[ag_num]  
                
                total_rewards.append(total_reward)
                
                last_rewards[ag_num] = total_reward  # Update the last recorded reward for the agent

                if ag_num == 0:
                    print("Total reward from rl-agent in iteration {} is {}".format(ep_num + 1,
                                                                                    total_rewards[-1]))
                else:
                    print("Total reward from ea-agent {} in iteration {} is {}".format(ag_num, ep_num + 1,
                                                                                       total_rewards[-1]))
            if self.training_mode:
                with open("ending_position.pkl", "wb") as f:
                    pickle.dump(self.agent[0].ending_position, f)
                with open("num_in_queue.pkl", "wb") as f:
                    pickle.dump(self.agent[0].num_in_queue, f)
                with open("total_rewards.pkl", "wb") as f:
                    pickle.dump(total_rewards, f)

                torch.save(self.agent[0].STATE_MEM, "STATE_MEM.pt")
                torch.save(self.agent[0].ACTION_MEM, "ACTION_MEM.pt")
                torch.save(self.agent[0].REWARD_MEM, "REWARD_MEM.pt")
                torch.save(self.agent[0].STATE2_MEM, "STATE2_MEM.pt")
                torch.save(self.agent[0].DONE_MEM, "DONE_MEM.pt")

                for ag_num in range(self.n_agents):
                    if ag_num == 0:
                        torch.save(self.agent[ag_num].local_net.state_dict(), "rl1.pt")
                        torch.save(self.agent[ag_num].target_net.state_dict(), "rl2.pt")
                    else:
                        torch.save(self.agent[ag_num].dqn.state_dict(), "ea{}.pt".format(ag_num))

            # Reward loggen
            log_reward.append(total_rewards)
            # print(log_reward)

            if (ep_num + 1) % 1 == 0:
                # average_rewards = np.mean(log_reward[-10:], axis=0)
                # average_rewards = np.delete(average_rewards, 0)  # Remove Agent 0 from average rewards
                # best_agents_ind = np.argsort(average_rewards)[-self.n_gutes_erbgut:][::-1] + 1

                lapp_rewards = np.delete(log_reward[-1], 0)
                best_agents_ind = np.argsort(lapp_rewards)[-self.n_gutes_erbgut:][::-1] + 1

                print("The best {} ea-agents of iteration {} were {}".format(self.n_gutes_erbgut, ep_num + 1,
                                                                             best_agents_ind))

                worst_agent_ind = np.argmin(lapp_rewards) + 1  # Find the index of the worst agent
                print("The worst ea-agent of iteration {} was {}".format(ep_num + 1, worst_agent_ind))

                for ag_num in range(self.n_agents):
                    if ag_num == worst_agent_ind:
                        self.agent[ag_num].dqn.load_state_dict(self.agent[0].target_net.state_dict())

                        print("Agent {} received weights from the rl-agent".format(ag_num))

                    if ag_num not in best_agents_ind:
                        if ag_num != 0:
                            if ag_num != worst_agent_ind:
                                # Crossover
                                selected_agents = random.sample(list(best_agents_ind), k=2)

                                mixed_weights = []
                                for param_idx, param in enumerate(self.agent[selected_agents[0]].dqn.parameters()):
                                    parent_weights_1 = list(self.agent[selected_agents[0]].dqn.parameters())[
                                        param_idx].clone()
                                    parent_weights_2 = list(self.agent[selected_agents[1]].dqn.parameters())[
                                        param_idx].clone()
                                    assert parent_weights_1.size() == parent_weights_2.size(), "Parameter dimensions do not match."

                                    new_weights = torch.empty_like(parent_weights_1)
                                    if (total_rewards[selected_agents[0]] + total_rewards[selected_agents[1]]) == 0:
                                        total_rewards_ratio = 0
                                    else:   
                                        total_rewards_ratio = total_rewards[selected_agents[0]] / (
                                                total_rewards[selected_agents[0]] + total_rewards[selected_agents[1]])
                                    mask = torch.rand(parent_weights_1.size()) < total_rewards_ratio
                                    new_weights[mask] = parent_weights_1[mask]
                                    new_weights[~mask] = parent_weights_2[~mask]

                                    mixed_weights.append(new_weights)

                                # Mutation
                                mutated_weights = []
                                for param in mixed_weights:
                                    noise = torch.empty_like(param).normal_(0, self.epsilon)
                                    mutated_weights.append(param + noise)

                                # Initialize the current agent with the mutated weights
                                state_dict = self.agent[ag_num].dqn.state_dict()
                                for idx, (name, param) in enumerate(state_dict.items()):
                                    state_dict[name] = mutated_weights[idx]
                                self.agent[ag_num].dqn.load_state_dict(state_dict)

                                print(
                                    "Agent {} is now the result of crossover and mutation with agents {} and {}, respectively".format(
                                        ag_num, selected_agents[0], selected_agents[1]))

        self.env.close()

        if self.it > 1:
            log_reward_array = np.array(log_reward)
            # print(log_reward_array)
            for i in range(self.n_agents):
                column = log_reward_array[:, i]
                # print(column)

                plt.title("Episodes trained vs. Average Rewards (per 100 eps)")
                plt.plot([0 for _ in range(100)] +
                         np.convolve(log_reward_array[:, i], np.ones((100,)) / 100, mode="valid").tolist())
            plt.show()

    def selektion(self):
        return
        # self.


#####################################################################################################################################################################################

if len(sys.argv) < 2:
    level = "SuperMarioBros-1-1-v0"
else:
    level = sys.argv[1]
  
_NeuroAgentManager = NeuroAgentManager(it=8000, pretrained=False, n_agents=11, training_mode=True, n_gutes_erbgut=3,
                                       epsilon=0.1, level=level)
_NeuroAgentManager.start_agents()
