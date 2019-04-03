import numpy as np
import wrsn as wr
from wrsn import *
import sys
import itertools
import matplotlib
from collections import defaultdict
import plotting
from plotting import *

ns=nsEnv(sensor_node)
ch=chEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
    


def q_learning(ns,ch, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    nsQ = defaultdict(lambda: np.zeros(ns.action_space.n))
    chQ = defaultdict(lambda: np.zeros(ch.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    ns_policy = make_epsilon_greedy_policy(nsQ, epsilon, ns.action_space.n)
    ch_policy = make_epsilon_greedy_policy(chQ, epsilon, ch.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 1 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        
        ns_state=ns.reset()
        
        ns_state=tuple(ns_state.flatten())
        # assert isinstance(ns_state,tuple)
        
        # One step in the environment
        
        for t in itertools.count():
            
            # Take a step
            ns_action_probs = ns_policy(ns_state)
            ns_action = np.random.choice(np.arange(len(ns_action_probs)), p=ns_action_probs)
            ns_next_state, ns_reward, done, _ = ns.step(ns_action)
            ns_next_state=tuple(ns_next_state.flatten())
            # Update statistics
            stats.episode_rewards[i_episode] += ns_reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            ns_best_next_action = np.argmax(nsQ[ns_next_state])    
            td_target = ns_reward + discount_factor * nsQ[ns_next_state][ns_best_next_action]
            td_delta = td_target - nsQ[ns_state][ns_action]
            nsQ[ns_state][ns_action] += alpha * td_delta
                
            if done:
                break
            ns_state = ns_next_state
            
            ch_state= tuple(wr.S[ns_action].flatten())
            ch_action_probs = ch_policy(ch_state)
            ch_action = np.random.choice(np.arange(len(ch_action_probs)), p=ch_action_probs)
            ch_next_state, ch_reward, done, _ = ns.step(ch_action)
            ch_next_state=tuple(ch_next_state.flatten())
            # TD Update
            ch_best_next_action = np.argmax(chQ[ch_next_state])    
            td_target = ch_reward + discount_factor * chQ[ch_next_state][ch_best_next_action]
            td_delta = td_target - chQ[ch_state][ch_action]
            chQ[ch_state][ch_action] += alpha * td_delta
                
            if done:
                break
            if t>500:
                break



    
    return nsQ,chQ, stats

nsQ,chQ, stats = q_learning(ns,ch, 1000)
plotting.plot_episode_stats(stats)