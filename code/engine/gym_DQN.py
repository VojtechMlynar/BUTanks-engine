"""Using BUTanks engine as custom gym environment for DQN (RAI 2021) DB

This is example of wrapping our BUTank engine on OpenAI gym module:
Created BUTanksGym class.

Attempt of training DQN was made but code needs a lot more time to refine.
Code runs but results aren't good, while only trying to move tank 1 into center.
Non DQN controlled tank has predefined inputs to win (with some delay).

Main causes of bad results:
    - Too much observations (states) = too much time for one episode.
    - Reward may not be good enough (now: closer to center = more reward).
    - All parameters needs refining.

Now in test trained DQN mode (just hit run).
Training must be done in FIXED_DT and more (see below).
To train uncomment agent.run() in main function and comment out agent.test().
"""

import gym
import numpy as np
import math
import random
import pygame
import engine

from collections import deque
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


config = engine.GameConfig()
config.NUM_OF_ANTENNAS = 8  # If not 8, manual edits in BUTanksGym needed
# CONFIG FOR TRAINING (if testing, comment out next few lines)
config.TARGET_FPS = 10000
config.RENDER_ALL_FRAMES = False  # False (To speed up)
config.TARGET_FRAME = 150  # Draw only every 150th frame
config.FIXED_DT = 1/25

# Constants
WIDTH, HEIGHT = 1000, 1000  # [px]
MAP_FILENAME = "map1.png"  # ../assets/maps (preset path)
TARGET_CAPTURE_TIME = 3  # [s]
TANK_SCALE = 1

# DQN parameters
LEARNING_RATE = 0.001
NUM_OF_ROUNDS = 100  # == EPISODES
DISCOUNT = 0.95  # Also know as gamma
EPSILON = 1.0  # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95  # (*= EPSILON_DECAY)
BATCH_SIZE = 32
TRAIN_START = 15

# Spawn lists
t1 = [(100, 100, 0)]
t2 = [(WIDTH - 100, HEIGHT - 100, 180)]
# Load predefined controls for FIXED_DT = 1/25 team 2 tank
op_tank_control = np.load('gym_DQN_predefined.npy')


# BUTanks engine gym custom environment wrap -----------------------------------
def decode_actions(multidiscrete_actions: np.ndarray):
    """Decode actions from binary to list for engine.Tank.input_AI().

    Returns: decoded_inputs = [phi_in, v_in, phi_rel_in, shoot]"""

    decoded_inputs = []
    for action in multidiscrete_actions:
        if action == 2:
            decoded_inputs.append(-1)
        else:
            decoded_inputs.append(action)
    return decoded_inputs


class BUTanksGym(gym.Env):
    """Wrap our custom environment around gym module. For standardized use.
    Few methods must be redefined:
        __init__()
        reset()
        step(action)
        render()

    Observation:
        Type: gym.spaces.Box()
        Num     Observation     Min     Max
        0       x               1       WIDTH
        1       y               1       HEIGHT
        2       phi             0       360
        3       phi_rel         0       360
        4       antenna 1       0       math.sqrt(WIDTH**2 + HEIGHT**2)
        n       antenna n       0       math.sqrt(WIDTH**2 + HEIGHT**2)

    Actions: phi_in, v_in, phi_rel_in, shoot
        Type: gym.spaces.MultiDiscrete() -> Discrete()
    """

    def __init__(self):
        """Init custom environment.

        Attributes to redefine: self.action_space, self.observation_space"""

        # Init engine.Game instance
        self.game = engine.Game(MAP_FILENAME, (WIDTH, HEIGHT), NUM_OF_ROUNDS,
                                game_config=config)
        # Actions we can take: phi+, phi-, v+, v-, phi_rel+, phi_rel-, shoot
        multi_discrete_actions = (3, 3, 3, 2)
        # Mapping multiDiscrete action space to Discrete
        self.action_mapping = tuple(np.ndindex(multi_discrete_actions))
        self.action_space = gym.spaces.Discrete(np.prod(multi_discrete_actions))
        # Observation space (states)
        low = np.array([1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0],
                       dtype=np.float32)
        high = np.array([WIDTH,
                         HEIGHT,
                         360,
                         360,
                         math.sqrt(WIDTH ** 2 + HEIGHT ** 2),
                         math.sqrt(WIDTH ** 2 + HEIGHT ** 2),
                         math.sqrt(WIDTH ** 2 + HEIGHT ** 2),
                         math.sqrt(WIDTH ** 2 + HEIGHT ** 2),
                         math.sqrt(WIDTH ** 2 + HEIGHT ** 2),
                         math.sqrt(WIDTH ** 2 + HEIGHT ** 2),
                         math.sqrt(WIDTH ** 2 + HEIGHT ** 2),
                         math.sqrt(WIDTH ** 2 + HEIGHT ** 2)],
                        dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.game.init_round(team_1_spawn_list=t1,
                             team_2_spawn_list=t2,
                             target_capture_time=TARGET_CAPTURE_TIME,
                             tank_scale=TANK_SCALE)
        # Start observation (state)
        ai_tank = self.game.team_1_list[0]
        self.state = np.array([ai_tank.x,
                               ai_tank.y,
                               ai_tank.phi,
                               ai_tank.phi_rel,
                               ai_tank.ant_distances[0],
                               ai_tank.ant_distances[1],
                               ai_tank.ant_distances[2],
                               ai_tank.ant_distances[3],
                               ai_tank.ant_distances[4],
                               ai_tank.ant_distances[5],
                               ai_tank.ant_distances[6],
                               ai_tank.ant_distances[7]],
                              dtype=np.float32)
        #  Set manual controls for opposing tank
        self.game.render_antennas_flag = True
        self.game.manual_input_flag = True
        self.game.team_2_list[0].manual_control_flag = True

    def reset(self):
        """Resets environment. Returns initial observation (state)."""

        self.game.init_round(team_1_spawn_list=t1,
                             team_2_spawn_list=t2,
                             target_capture_time=TARGET_CAPTURE_TIME,
                             tank_scale=TANK_SCALE)
        # Debug mode example:
        self.game.render_antennas_flag = True
        self.game.manual_input_flag = True
        self.game.team_2_list[0].manual_control_flag = True

        # Reset init state
        ai_tank = self.game.team_1_list[0]
        self.state = np.array([ai_tank.x,
                               ai_tank.y,
                               ai_tank.phi,
                               ai_tank.phi_rel,
                               ai_tank.ant_distances[0],
                               ai_tank.ant_distances[1],
                               ai_tank.ant_distances[2],
                               ai_tank.ant_distances[3],
                               ai_tank.ant_distances[4],
                               ai_tank.ant_distances[5],
                               ai_tank.ant_distances[6],
                               ai_tank.ant_distances[7]],
                              dtype=np.float32)
        # Reset done eval
        return self.state

    def step(self, action):
        """Performs simulation step based on action."""

        self.game.get_delta_time()
        self.game.check_and_handle_events()
        # -------------------------------------------------------
        # INPUT preferably here (game.inputAI())
        multidiscrete_actions = self.action_mapping[action]
        input_AI_action = decode_actions(multidiscrete_actions)
        self.game.team_1_list[0].input_AI(input_AI_action)
        # -------------------------------------------------------
        self.game.update()
        self.game.check_state()

        ai_tank = self.game.team_1_list[0]
        self.state = np.array([ai_tank.x,
                               ai_tank.y,
                               ai_tank.phi,
                               ai_tank.phi_rel,
                               ai_tank.ant_distances[0],
                               ai_tank.ant_distances[1],
                               ai_tank.ant_distances[2],
                               ai_tank.ant_distances[3],
                               ai_tank.ant_distances[4],
                               ai_tank.ant_distances[5],
                               ai_tank.ant_distances[6],
                               ai_tank.ant_distances[7]],
                              dtype=np.float32)
        # Calculate reward
        capture_area_center = (WIDTH/2, HEIGHT/2)
        if ai_tank.capturing_flag:
            reward = 10000
        else:
            init_distance = math.sqrt((t1[0][0] - capture_area_center[0]) ** 2 +
                                      (t1[0][1] - capture_area_center[1]) ** 2)
            calc_distance = math.sqrt((ai_tank.x - capture_area_center[0]) ** 2 +
                                      (ai_tank.y - capture_area_center[1]) ** 2)
            reward = (init_distance - calc_distance)/2
            print(f'reward: {reward}')

        # Check done eval
        if not self.game.round_run_flag:
            done = True
        else:
            done = False
        # Setup placeholder for info
        info = {}
        # Return step information
        return self.state, reward, done, info

    def render(self, mode='Human'):
        """Render frame method."""

        self.game.draw_background()
        # Place to draw under tanks:
        self.game.draw_tanks()
        # Place to draw on top of tanks:
        self.game.update_frame()


# DQN SETUP --------------------------------------------------------------------
class DQNAgent:
    """DQN agent."""

    def __init__(self):
        self.env = BUTanksGym()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = NUM_OF_ROUNDS
        self.memory = deque(maxlen=1000)

        self.discount = DISCOUNT
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.batch_size = BATCH_SIZE
        self.train_start = TRAIN_START

        self.model = self._build_DQN_model()

    def _build_DQN_model(self):
        """Build model.

        Regression NN for Q table approximation.
        """

        model = Sequential()

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        model.summary()
        return model

    def memorize(self, state, action, reward, next_state, done):
        """Saves to memory and decreases epsilon."""

        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        """Choose exploration/exploitation and return action."""

        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def train_dqn(self):
        """Train DQN."""

        # Not enough data in memory
        if len(self.memory) < self.train_start:
            return
        # Choose random samples from memory
        minibatch = random.sample(self.memory, min(len(self.memory),
                                                   self.batch_size))
        # Input preparation
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []
        # Create input and output
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        # States prediction (target is array of q values)
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        # Output customization
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount * \
                                       (np.amax(target_next[i]))
        # Train NN using state
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        """Load DQN model."""

        self.model = load_model(name)

    def save(self, name):
        """Save DQN model."""

        self.model.save(name)

    def run_training(self):
        """"""

        try:
            for e in range(self.EPISODES):
                if self.env.game.quit_flag:
                    break
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                i = 0
                while not done:
                    # Predefined controls for opposite tank
                    op_tank = self.env.game.team_2_list[0]
                    op_tank.input_AI(op_tank_control[i])

                    action = self.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    if not (done or self.env.game.quit_flag):
                        reward = reward
                        self.env.render()
                    else:
                        if self.env.game.win_team == 1:
                            reward = reward + 100
                        elif self.env.game.win_team == 0:
                            reward = reward
                        else:
                            reward = -100
                    self.memorize(state, action, reward, next_state, done)
                    state = next_state
                    i += 1
                    if done:
                        str1 = f'Episode {e}/{self.EPISODES}, reward: {reward}'
                        str2 = f', epsilon {self.epsilon}'
                        print(str1 + str2)
                self.train_dqn()
        finally:
            pygame.quit()
            print("Final win list (0 = Tie): ", self.env.game.win_list)
            print("Quitting.")
            self.save("gym_DQN_trained.h5")

    def test(self):
        self.load("gym_DQN_trained.h5")
        for e in range(10):
            if self.env.game.quit_flag:
                break
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                if not (done or self.env.game.quit_flag):
                    self.env.render()
                i += 1
                if done:
                    str1 = f'Episode {e}/{self.EPISODES}, reward: {reward}'
                    str2 = f', epsilon {self.epsilon}'
                    print(str1 + str2)


def main():
    agent = DQNAgent()
    agent.run_training()
    # agent.test()


if __name__ == "__main__":
    main()
