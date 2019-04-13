import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
import wrsn2 as wr
from wrsn2 import *
import plotting
from plotting import *
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

from sklearn.externals import joblib

env=wrsn(sensor_node)

scaler = joblib.load("scaler.model",'r')


featurizer=joblib.load("fearture.model",'r')

class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])

    

def q_learning(env, estimator, num_episodes):
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_transbag=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):
    
        # The policy we're following
        
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            
            
            q_values = estimator.predict(state)
            best_action = np.argmax(q_values)
            # Take a step
            next_state, reward, done, data_overflow = env.step(best_action)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward 
            stats.episode_lengths[i_episode] = t
            stats.episode_transbag[i_episode]+=data_overflow
            # TD Update
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
                
            if done:
                break
            if t>=200:
                break
            state = next_state
    
    return stats

estimator=joblib.load("estimator.model",'r')
stats = q_learning(env, estimator, 50)

plotting.plot_episode_stats(stats)

