import numpy as np
import argparse, sys
from os.path import isfile
import gym
import time
import pickle
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

## setup
keras.backend.set_image_data_format('channels_last')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Train or play. If play mode, model weights file must exist. Default: train', type=str, default='train')
parser.add_argument('--file_name', help='File to save model weights to. Make sure to change this parameter if training/comparing multiple models. Default: pong_model.h5', type=str, default='pong_model.h5')
parser.add_argument('--reset_weights', help='Whether or not to overwrite model weights file. Default: True', type=bool, default=True)
parser.add_argument('--learning_rate', help='Learning rate of policy gradient updates. Default: 0.2', type=float, default=0.2)
parser.add_argument('--gamma', help='Discount rate for episode rewards. Default: 0.99', type=float, default=0.99)
parser.add_argument('--n_dense_nodes', help='Number of nodes in Dense layer. Default: 512', type=int, default=512)
parser.add_argument('--score_goal', help='The model will be trained until it reaches this score on average for n_games_goal. Default: 21', type=float, default=21)
parser.add_argument('--n_games_goal', help='Number of episodes used to track average episode scores. Default: 25', type=int, default=25)
parser.add_argument('--render', help='Whether or not to display games as model trains. Default: False', type=bool, default=False)
parser.add_argument('--log_file', help='File to log episode results to. Default: training_log.txt', type=str, default='training_log.txt')
parser.add_argument('--n_games', help='If mode is play, the number of games to play. Default: 25', type=int, default=25)
parser.add_argument('--render_speed', help='If mode is play, speed of rendering. Options: 1 (fast), 2 (medium), 3 (slow). Default: 1', type=int, default=1)
args = parser.parse_args()

mode = args.mode
file_name = args.file_name
reset_weights = args.reset_weights
learning_rate = args.learning_rate
gamma = args.gamma
n_dense_nodes = args.n_dense_nodes
score_goal = args.score_goal
n_games_goal = args.n_games_goal
render = args.score_goal
log_file = args.log_file
n_games = args.n_games
render_speed = args.render_speed

def process_observation(image):
    '''
      input: raw Pong observation of shape (210, 160, 3) 
      output: cropped subsampled observation of shape (78, 65) in black and white
    '''
    ## crop and subsample
    image = image[35:190:2, 15:145:2, 0]
    ## set colors to black or white
    image[np.logical_or(image == 144, image == 109)] = 0
    image[image != 0] = 1
    return image.astype(np.float)
 
def process_rewards(rewards_list, gamma=0.99):
    '''
      input: list of rewards from a game, discount rate gamma
      output: discounted rewards
    '''
    ## create episode buckets that end in a non-zero reward
    reward_trajectories = []
    current_reward_trajectory = []
    for i in rewards_list:
      current_reward_trajectory.append(i)
      if i!=0:
        reward_trajectories.append(current_reward_trajectory)
        current_reward_trajectory = []
    
    ## calculate discounted rewards for each episode
    discounted_rewards = []
    for trajectory in reward_trajectories:
      terminal_reward = trajectory[-1]
      for i in reversed(range(len(trajectory))):
        current_reward = trajectory[i]
        discounted_rewards.append(current_reward + terminal_reward*gamma**i)

    ## convert to numpy array and reshape (length,) --> (length, 1)
    discounted_rewards = np.array(discounted_rewards).reshape(-1,1)
    return discounted_rewards

def init_model():
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=((78, 65, 2))))
    model.add(Dense(n_dense_nodes, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dense(number_of_actions, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    ## load model weights if file exists
    if mode == 'train':
      if isfile(file_name) and not reset_weights:
        model.load_weights(file_name)
    if mode == 'play':
      if isfile(file_name):
        model.load_weights(file_name)
      else:
        raise Exception(f"Model weights file {file_name} not found. Please provide a weights file or train model first.")
    return model

## action space parameters
number_of_actions = 3
action_space = [1,2,3]

## gym environment initialization
env = gym.make("PongDeterministic-v4")
observation = env.reset()
previous_observation = process_observation(observation)


if mode == 'train':
  ## setup
  states, action_probability_gradients, rewards, action_probabilities, reward_sums = [], [], [], [], []
  reward_sum = 0
  game_number = 0
  model = init_model()

  ## training loop
  train = True
  while train:
      ## get current state by combing current observation and previous observation
      current_observation = process_observation(observation)
      state = np.hstack([current_observation, previous_observation])
      previous_observation = current_observation
      
      ## predict action probabilities and select chosen action
      action_probabilities_ = model(state.reshape(1, 78, 65, 2)).numpy()[0, :]    
      action_index = np.random.choice(number_of_actions, p=action_probabilities_)
      action = action_space[action_index]
  
      ## execute chosen action
      observation, reward, done, info = env.step(action)

      ## calculate action probability gradients
      pseudo_action_probabilities = np.zeros(number_of_actions)
      pseudo_action_probabilities[action_index] = 1
      
      ## remember state, action probabilities, action probability gradients, rewards
      reward_sum += reward
      states.append(state)
      action_probabilities.append(action_probabilities_)
      action_probability_gradients.append(pseudo_action_probabilities - action_probabilities_)
      rewards.append(reward)

      ## check if game finished
      if done:
          ## increment game 
          game_number += 1
  
          ## keep track of previous n_games_goal ganes
          reward_sums.append(reward_sum)
          if len(reward_sums) > n_games_goal:
              reward_sums.pop(0)
  
          ## log game results
          game_results = 'Game %d - Score: %f , Running Average Score: %f' % (
              game_number, reward_sum, np.mean(reward_sums))
          print(game_results)
          with open(log_file, "a") as f:
            f.write(game_results + '\n')
          if np.mean(reward_sums) >= score_goal:
            print(f"Training complete. Model has reached score goal of {score_goal} for {n_games_goal} consecutive games.")
            train = False
            break

          ## handle reward propagation
          action_probability_gradients = np.vstack(action_probability_gradients)
          rewards = process_rewards(rewards)
  
          ## construct training data
          X = np.vstack(states).reshape(-1, 78, 65, 2)
          y = action_probabilities + learning_rate * rewards * action_probability_gradients

          ## train model
          model.train_on_batch(X, y)
          model.save_weights(file_name)

          ## reset for next game
          states, action_probability_gradients, rewards, action_probabilities = [], [], [], []
          reward_sum = 0
          obs = env.reset()

if mode == 'play':
  for game in range(n_games):
    done = False
    while not done: 
      env.render()
      time.sleep((render_speed - 1)/2)
      current_observation = process_observation(observation)
      state = np.hstack([current_observation, previous_observation])
      previous_observation = current_observation
      
      action_probabilities_ = model(state.reshape(1, 78, 65, 2)).numpy()[0, :]    
      action_index = np.random.choice(number_of_actions, p=action_probabilities_)
      action = action_space[action_index]
  
      observation, reward, done, info = env.step(action)
    observation = env.reset()
    previous_observation = process_observation(observation)