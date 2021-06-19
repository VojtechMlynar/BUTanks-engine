"""BUTanks engine with applied actor-critic reinforcement learning (RAI 2021)

This script trains two agents against themselves. Semi-graphic mode is highly
recommended, as it takes ~100 rounds to get at least semi-intelligent agents.
You can use manual control to try and compete against trained bot. You can
even try and change the map to evaluate agents generalization (spoiler alert
- not so great)

requires: pygame, tensorflow, tensorflow_probability, pyastar2d (might 
require to install manually - just copy Github repo and follow instructions)

Author: Vojtech Mlynar
Date: 18.06.2021
"""

import os
from pathlib import Path
import engine
import tanks_utility as tu
import numpy as np
import random
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers


def get_game_state(game: engine.Game, masks: tu.ArenaMasks, team):
    """Get current game state

    Creates masks.sized matrix filled with 0.5. 9 pixels respective to friendly
    tank position are set to 1 and 9 pixels surrounding adversary tank position
    are set to 0.

    arguments:
        game -- engine.Game object handle
        masks -- ArenaMasks object handle
        team -- number of friendly team .. 1/2
    
    returns:
        mask.size shaped numpy array
    """
    scl = masks.res_scale
    state = np.ones((masks.size))*0.5
    if team == 1:
        x,y = game.team_1_list[0].x, game.team_1_list[0].y
        x_e, y_e = game.team_2_list[0].x, game.team_2_list[0].y
    else:
        x,y = game.team_2_list[0].x, game.team_2_list[0].y
        x_e, y_e = game.team_1_list[0].x, game.team_1_list[0].y
    
    # Convert high-res cooridates to low-res
    x, y = round(x/scl[0]), round(y/scl[1]) 
    x_e, y_e = round(x_e/scl[0]), round(y_e/scl[1])
    
    dirs = np.array([[0, 0],[-1, 0],[1, 0],[0,-1],[0, 1],
                     [1, 1],[1,-1],[-1, 1],[-1,-1]], dtype=int)
    limit = masks.size
    for d in dirs:
        xd, yd = x+d[0], y+d[1]
        if(xd>=0) & (xd<limit[0]) & (yd>=0) & (yd<limit[1]):
            state[xd,yd] = 1
        xd, yd = x_e+d[0], y_e+d[1]
        if(xd>=0) & (xd<limit[0]) & (yd>=0) & (yd<limit[1]):
            state[xd,yd] = 0
    return state

def get_reward(game:engine.Game, valid_click, team):
    """Get reward value for specified team
    
    Returns reward based on weights specified lower.

    arguments:
        game -- engine.Game object handle
        valid_click -- bool whether positions specified by AI is valid move
        team -- friendly team number (1/2)
    """
    W_HEALTH_SELF = 0.1
    W_HEALTH_ENEMY = 1
    W_CAPTURE_SELF = 5
    W_CAPTURE_ENEMY = 1
    W_CLICK_INVALID = -0.1
    W_CLICK_VALID = 0
    MAX_HEALTH = 5

    max_time = game.target_capture_time
    t_1 = game.team_1_captured_time/max_time
    t_2 = game.team_2_captured_time/max_time
    h_1 = game.team_1_list[0].health/MAX_HEALTH
    h_2 = game.team_2_list[0].health/MAX_HEALTH

    if team == 1:
        reward = h_1*W_HEALTH_SELF + t_1*W_CAPTURE_SELF \
                - h_2*W_HEALTH_ENEMY - t_2*W_CAPTURE_ENEMY \
                + (MAX_HEALTH-h_2)*W_HEALTH_ENEMY
    else:
        reward = h_2*W_HEALTH_SELF + t_2*W_CAPTURE_SELF \
                - h_1*W_HEALTH_ENEMY - t_1*W_CAPTURE_ENEMY \
                + (MAX_HEALTH-h_1)*W_HEALTH_ENEMY
    
    # Adjust reward if "click" (position where NN wants the tank to move) 
    # is valid
    if valid_click is False:
        reward += W_CLICK_INVALID
    else:
        reward += W_CLICK_VALID

    return reward

def check_valid_click(arena: tu.ArenaMasks, click):
    """Check if AI specified positions is valid move
    
    Returns True if "click" is within map and specified positions is 
    reachable by A* path planning (obstacles_dil[x,y] == 0)

    arguments:
        arena -- ArenaMasks object handle
        click -- [X,Y] format 
    """
    x, y = round(click[0]), round(click[1])
    boundary = arena.size

    if ((x>0) & (y>0) & (x<(boundary[0]-1)) & (y<(boundary[1]-1))):
        if(arena.obstacles_dil[x,y] == 0):
            return True
        else:
            return False
    else:
        return False

class ActorCriticAgent:
    """Agent class for Agent-critic network"""
    def __init__(self, static_state, learn_rate=1e-5):
        """Agent initialization witg static state and learn rate
        
        arguments:
            static_state -- low_res image_width x image_height x 3 numpy array
                         -- layer[:,:,0] should be obstacles
                         -- layer[:,:,1] should be capture area pixels only
                         -- layer[:,:,2] contains positions of tanks, this is
                            only layer that is changing with time
            learn_rate -- Adam optimizer learning rate, keep this small (~1e-5)
        """
        self.static_state = static_state
        self.mem_logprob = []
        self.mem_reward = []
        self.mem_critic = []
        self.memory_done = []
        self.gamma = 0.99    # discount rate
        self.learning_rate = learn_rate
        self.optimizer = keras.optimizers.Adam(learning_rate=learn_rate)
        self.huber_loss = keras.losses.Huber()

        self.model = self._build_model()

    def _build_model(self):
        """Build actor-critic network"""
        inputs = layers.Input(shape=(100,100,3,))

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(16, 5, strides=5, activation="relu")(inputs)
        layer2 = layers.Conv2D(32, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer2)
        layer4 = layers.Flatten()(layer3)
        layer5 = layers.Dense(400, activation="relu")(layer4)
        layer6 = layers.Dense(200, activation="relu")(layer5)
        mu = layers.Dense(2,activation="tanh")(layer6)
        mu = (mu+1)/2 #Get output initialized on 0.5 and limited to 0..1
        sigma = layers.Dense(2, activation="softplus")(layer6)+1e-5

        critic = layers.Dense(1)(layer6) # Critic output layer
        model = keras.Model(inputs=inputs, outputs=[mu, sigma, critic])

        return model

    def memorize(self, logprob, critic, reward):
        """Append provided values to memory attributes"""
        self.mem_logprob.append(logprob)
        self.mem_critic.append(critic[0, 0])
        self.mem_reward.append(reward)

    def learn(self, tape):
        """Update the net weights based on recorded history
        
        tape -- tensorflow gradient tape handle
             -- note: When using more than 1 agent, use persistent tape
        """
        returns = []
        discounted_sum = 0
        for r in self.mem_reward[::-1]:
            discounted_sum = r + self.gamma*discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize? Seems to work better without it
        returns = np.array(returns)
        #returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        
        # Calculating loss values to update our network
        history = zip(self.mem_logprob, self.mem_critic, returns)
        actor_losses = []
        critic_losses = []
        for prob, value, ret in history:
            # At this point in history, the critic estimated that we would get
            # a total reward = `value` in the future. We took an action with
            # log probability of `log_prob` and ended up recieving a total
            # reward = `ret`. The actor must be updated so that it predicts an
            # action that leads to high rewards (compared to critic's estimate)
            # with high probability.
            diff = ret - value
            actor_losses.append(-prob * diff)  # actor loss
            # The critic must be updated so that it predicts a better estimate
            # of the future rewards.
            critic_losses.append(
                self.huber_loss(tf.expand_dims(self.mem_critic, 0),
                                tf.expand_dims(ret, 0))
            )
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        
        # Clear the loss and reward history
        self.mem_logprob.clear()
        self.mem_critic.clear()
        self.mem_reward.clear()

    def act(self, state):
        """Get net output """
        state_complete = self.static_state
        state_complete[:,:,2] = state
        state_complete = tf.convert_to_tensor(state_complete)
        state_complete = tf.expand_dims(state_complete, 0)
        mu, sigma, critic_value = self.model(state_complete)

        return mu, sigma, critic_value  # returns action

    def get_action_value(self,state):
        """Get action value from net output
        
        state -- image shaped (w,h,1) numpy array with tank positions
        """
        # Get mu and sigma predictions
        mu, sigma, critic_value = self.act(state)
        # Create normal distribution with predicted parameters
        n=tfp.distributions.Normal(loc=mu,scale=sigma)
        action = n.sample() #get sample from distribution
        prob = n.prob(action) #get probability of selected action

        # Saturate probability to 1 - larger "probabilites" cause net to drift
        # away from already found solutions to nonsensical ones
        prob = tf.minimum(tf.constant([1.0, 1.0]),prob)
        
        # Make logarithmic probability
        log_prob = tf.math.log(prob)
        
        # Convert 0..1 action to 0-99 discrete values - coordinates/"click"
        xy = np.squeeze(np.round(np.clip(action*100,0,99)))
        return xy.astype(int), log_prob, critic_value

    def save_model(self, filepath):
        """Save trained net to "filepath" folder"""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load trained net to "filepath" folder"""
        self.model = keras.models.load_model(filepath)

    def load_weights(self, name):
        """Load net weights"""
        self.model.load_weights(name)

    def save_weights(self, name):
        """Save net weights only"""
        self.model.save_weights(name)

# ------------------------
# MAIN LOOP
# ------------------------

eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
gamma = 0.99

WIDTH, HEIGHT = 1000, 1000
MAP_FILENAME = "map2.png" # ../assets/maps (preset path)
TARGET_CAPTURE_TIME = 5  # [s]
NUM_OF_ROUNDS = 5000
TANK_SCALE = 1
T_UPDATE = 0.5
BATCH = 5
# You can use these pretrained nets (trained on map2.png):
NET_FILENAME1 = "Clicker_blue2"
NET_FILENAME2 = "Clicker_red2"

p = Path(__file__).parents[1]
NET1_DIR = os.path.join(p, "assets", "trained_nets", NET_FILENAME1)
NET2_DIR = os.path.join(p, "assets", "trained_nets", NET_FILENAME2)

def main():
    # Config
    config = engine.GameConfig()
    config.MAP_BACKGROUND_COLOR = (100, 100, 100)
    # Create engine.Game class instance
    game = engine.Game(MAP_FILENAME, (WIDTH, HEIGHT), NUM_OF_ROUNDS,
                       game_config=config)
    
    t1 = [(100,500,90)]
    t2 = [(WIDTH-100, HEIGHT-500, 270)]
    t1yrng = [250, 750] # Set varying spawn Y location
    t2yrng = t1yrng
    # USER INIT
    
    masks = tu.ArenaMasks(game.arena, nav_margin=35)
    static_state = np.zeros((100,100,3))
    static_state[:,:,0] = masks.obstacles_bin
    static_state[:,:,1] = masks.capture_area_mask
    # Create actor-critic agents
    agent1 = ActorCriticAgent(static_state,5e-6)
    agent2 = ActorCriticAgent(static_state,5e-6)
    # Load pretrained nets (you can comment these out to start fresh)
    agent1.load_model(NET1_DIR)
    agent2.load_model(NET2_DIR)
    running_reward = 0

    # Main loop
    while not game.quit_flag:
        t1 = [(100,round(random.uniform(t1yrng[0],t1yrng[1])), 180)]
        t2 = [(WIDTH-100,round(random.uniform(t2yrng[0],t2yrng[1])) , 180)]

        game.init_round(team_1_spawn_list=t1,
                        team_2_spawn_list=t2,
                        target_capture_time=5,
                        tank_scale=1)
        # Set flags
        game.render_antennas_flag = False 
        game.manual_input_flag = True
        game.team_2_list[0].manual_control_flag = True
        # Create controller objects
        static_state[:,:,2] = get_game_state(game, masks, 1)
        red_controller = tu.AIController(masks, 
                                          game.team_2_list[0],
                                          game.team_1_list[0],
                                          toothless=False)
        blue_controller = tu.AIController(masks, 
                                           game.team_1_list[0],
                                           game.team_2_list[0])
        state1 = static_state[:,:,2]
        state2 = static_state[:,:,2]
        episode_reward = 0
        tic = 1
        round_tic = 0
        valid_click1 = False
        valid_click2 = False
        e = 0
        
        # Individual round loop
        while game.round_run_flag:
            with tf.GradientTape(persistent=True) as tape:
                game.get_delta_time()
                game.check_and_handle_events()
                # -------------------------------------------------------
                # INPUT game.inputAI()
                tic += game.dt # To time controller updates
                round_tic += game.dt # For timeout
                if tic > T_UPDATE:
                    # Get tank positions and compile appropriate matrices
                    state1 = get_game_state(game, masks, 1)
                    state2 = get_game_state(game, masks, 2)
                    # Get predictions/actions with agents
                    xy1, log_prob1, critic_value1 = agent1.get_action_value(state1)
                    xy2, log_prob2, critic_value2 = agent2.get_action_value(state2)
                    # if xy action was plausible, plan path and set controller
                    if check_valid_click(masks, xy1) is True:
                        tf.print(xy1, log_prob1, critic_value1)
                        path = blue_controller.plan_path_astar(
                                                    masks.nav_matrix, xy1)
                        blue_controller.set_waypoints(path[1:,:])
                        valid_click1 = True
                    else:
                        valid_click1 = False

                    if check_valid_click(masks, xy2) is True:
                        path = red_controller.plan_path_astar(
                                                    masks.nav_matrix, xy2)
                        red_controller.set_waypoints(path[1:,:])
                        valid_click2 = True
                    else:
                        valid_click2 = False
                # Push action values to the tank itself
                blue_controller.controls_output()
                red_controller.controls_output() #Comment this for manual control
                # -------------------------------------------------------
                game.update()
                game.draw_background()
                # Place to draw under tanks
                game.draw_tanks()
                # Place to draw on top of tanks
                game.update_frame()
                game.check_state()
                # -------------------------------------------------------
                #  OUTPUT
                if(round_tic>30): # Round timeout -- tanks got stuck probably
                    game._win_team=0 # end with tie
                if tic > T_UPDATE:
                    tic = 0
                    #Get rewards based on last actions
                    reward1 = get_reward(game, valid_click1, team=1)
                    reward2 = get_reward(game, valid_click2, team=2)
                    #Append probabilites, critic values and rewards to memory
                    agent1.memorize(log_prob1, critic_value1, reward1)
                    agent2.memorize(log_prob2, critic_value2, reward2)
                    e+=1
                    if(e>BATCH):
                        agent1.learn(tape) #Comment these out 
                        agent2.learn(tape) #to stop agents updating
                        e=0
                    episode_reward += reward1
                # -------------------------------------------------------
                # print("FPS: ", (1/(game.last_millis/1000)))
            #Episode finished
        running_reward = 0.05*episode_reward+(1-0.05)*running_reward
        print("Running reward:", running_reward)
    agent1.load_model(NET1_DIR)
    agent2.load_model(NET2_DIR)

if __name__ == "__main__":
    main()
