import numpy as np
import wrsn2 as wr
from wrsn2 import *
import sys
import itertools
import matplotlib
from collections import defaultdict
import plotting
from plotting import *
import json

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

def q_learning(env, num_episodes,explor_epi, discount_factor=0.99, alpha=0.25, epsilon=0.15):
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
    
  
    
    # with open("nsQ_json.json","r") as json_file:
    #     nsQ1=json.load(json_file)
    # json_file.close()
    # for key in list(nsQ1.keys()):
    #     nsQ1[tuple(np.array(key.split()))] = np.array(nsQ1[key].split())
    #     del nsQ1[key]

    # with open("chQ_json.json","r") as json_file:
    #     chQ1=json.load(json_file)
    # json_file.close()
    # for key in list(chQ1.keys()):
    #     chQ1[tuple(np.array(key.split()))] = np.array(chQ1[key].split())
    #     del chQ1[key]
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # for k,v in nsQ1.items():
    #     nsQ[k]=v
    # for k,v in chQ1.items():
    #     chQ[k]=v
    # for k,v in chQ1.items():
    #     print(k,v)
    #     break
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_transbag=np.zeros(num_episodes))    
    
    # The policy we're following
    
    

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if(i_episode<=explor_epi):
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        else:
            policy = make_epsilon_greedy_policy(Q, 0.0, env.action_space.n)
        sys.stdout.flush()
        last_reward = stats.episode_transbag[i_episode - 1]
        # Reset the environment and pick the first action
        
        state=env.reset_test()
        
        # One step in the environment
        
        for t in itertools.count():
            #print(ns_state)
            #print(type(ns_state))
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, data_overflow = env.step(action)        
            # Update statistics
          
            stats.episode_rewards[i_episode] += reward 
            stats.episode_lengths[i_episode] = t
            stats.episode_transbag[i_episode]+=data_overflow
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            state = next_state
            
            
            if t>1000:
                break

        print("\r@ Episode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")

    
    return Q, stats

env=wrsn(sensor_node)
print(((queue_len+1)*(battle_level+1)*distance_level)**sensor_node)
Q, stats = q_learning(env, 3000,2900)
for i in range(len(stats.episode_rewards)):
    stats.episode_rewards[i]/=1

plotting.plot_episode_stats(stats)

ostate=0
for key in list(Q.keys()):
    for a in range(env.action_space.n):
        if np.abs(Q[key][a])<1e-3:
            ostate+=1

    Q[str(list(key))] = str(Q[key])
    del Q[key]
print(ostate)
print(len(Q)*env.action_space.n)
print()
with open("version2/Q_json.json","w") as json_file:
    json_file.writelines(json.dumps(Q,sort_keys=True, indent=4, separators=(',', ': ')))
        
json_file.close()


