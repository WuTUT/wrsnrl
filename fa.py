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
env=wrsn(sensor_node)
from sklearn.externals import joblib
observation_examples = np.array([tuple(env.observation_space.sample().flatten()) for x in range(10000)])
#print(observation_examples.shape)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

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

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon_start=1.0,epsilon_end=0.1,epsilon_decay_steps=400):
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)      
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_transbag=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):
    
        # The policy we're following
        #policy = make_epsilon_greedy_policy(estimator, epsilons[min(i_episode,epsilon_decay_steps-1)], env.action_space.n)
        policy = make_epsilon_greedy_policy(estimator, 0.1, env.action_space.n)
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_transbag = stats.episode_transbag[i_episode - 1]
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset_test()
        
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # Take a step
            next_state, reward, done, data_overflow = env.step(action)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward 
            stats.episode_lengths[i_episode] = t
            stats.episode_transbag[i_episode]+=data_overflow
            # TD Update
            q_values_next = estimator.predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
            

            # Update the function approximator using our target
            estimator.update(state, action, td_target)
            
            print("\rStep {} @ Episode {}/{} r({}) t({})".format(t, i_episode + 1, num_episodes, last_reward,last_transbag), end="")
                
            if done:
                break
            if t>=200:
                break
            state = next_state
    
    return stats

estimator = Estimator()
stats = q_learning(env, estimator, 100)

plotting.plot_episode_stats(stats)

joblib.dump(featurizer,"fearture.model")
joblib.dump(estimator,"estimator.model")
joblib.dump(scaler,"scaler.model")

def fa_test(env, estimator, num_episodes):
    stats_test = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_transbag=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):
    
        # The policy we're following      
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats_test.episode_transbag[i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset_test()
        
        # One step in the environment
        for t in itertools.count():

            q_values = estimator.predict(state)
            best_action = np.argmax(q_values)
            next_state, reward, done, data_overflow = env.step(best_action)
            
            # Update statistics
            stats_test.episode_rewards[i_episode] += reward 
            stats_test.episode_lengths[i_episode] = t
            stats_test.episode_transbag[i_episode]+=data_overflow
            if done:
                break
            if t>=1000:
                break
            state = next_state
        print("\r@ Episode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")

    return stats_test
stats_test=fa_test(env,estimator,10)
plotting.plot_episode_stats(stats_test)
sys.stdout.flush()
print(np.mean(stats_test.episode_transbag))