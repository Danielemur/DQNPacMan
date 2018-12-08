import gym
import numpy as np
import random
from time import time
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from collections import deque

# Referenced https://keon.io/deep-q-learning/ for starting point

# Deep Q-learning Agent
class Player:
    def __init__(self, env):
        self.env = env
        self.state_size = len(env.observation_space.high) # number of state parameters
        self.action_size = env.action_space.n # number of possible actions
        self.memory = deque(maxlen=10000) # memory stores max of 10000 events
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001 # for the neural net
        self.model = self._build_model() # untrained neural net
        
    def _build_model(self):
        # Create a neural network
        model = Sequential()

        # Tweak number of neurons in layers to account for larger state size
        model.add(Dense(96, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Act in an epsilon greedy manner
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample() # Choose action randomly
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # Greedily choose "best" action

    def act_greedy(self, state):
        # Act in a greedy manner after environment is solved
        return np.argmax(self.model.predict(state)[0]) 
    
    def replay(self, batch_size):
        # Learn from past experiences
        if batch_size > len(self.memory): # Not enough memories yet
            return

        # Pick a random set of experiences to learn from
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            # If we're at a terminal state, no need to look at next state
            if not done:
                # Standard value function
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target # alpha = 1 in this agent
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Gradually decrease exploration rate over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_episode(agent):
    env = agent.env
    
    # Get initial state
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    R = 0
    steps = 0

    done = False
    while not done:
        #env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        R += reward
        steps += 1
        if done:
            return (R, steps)

# Initialize gym environment and the agent
env = gym.make('MsPacman-ram-v0')
env = gym.wrappers.Monitor(env, './videos/' + str(time()) + '/',
                           video_callable=lambda episode_id: episode_id%10==0)
agent = Player(env)
episodes = 10000
rewards = deque()

# Build Memory
print("Building memory:")
for i in range(50):
    print('Episode {}/{}'.format(i + 1, 50), flush=True)
    run_episode(agent)
print('')

# Learn
print('Begin learning:')
for e in range(episodes):
    # Run the episode
    (R, steps) = run_episode(agent)

    # Calculate average reward
    rewards.append(R)
    avg = np.average(rewards)

    print("episode: {0}/{1}, {2} steps, reward: {3}, recent average: {4:.2f}"
          .format(e+1, episodes, steps, R, avg), flush=True)

#    if e >= 100 and np.average(rewards) > -180:
#        print("Environment Solved")
#        break

    # Review random subset of memories
    agent.replay(1000)

# Demo trained agent
env = gym.wrappers.Monitor(env, './videos/' + str(time()) + '_trained/',
                           video_callable=lambda episode_id: True)

frames = []
for i in range(3):
    obs = env.reset()
    obs = obs.reshape(1, agent.state_size)
    done = False
    R = 0
    t = 0
    while not done:
        frames.append(env.render(mode = 'rgb_array'))
        action = agent.act_greedy(obs)
        obs, r, done, _ = env.step(action)
        obs = obs.reshape(1,agent.state_size)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
env.render()


# Save model
agent.model.save(str(time()) + '_trained.h5')
