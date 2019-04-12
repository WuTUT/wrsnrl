import numpy as np
import wrsn as wr
from wrsn import *
import sys
import itertools
import matplotlib
from collections import defaultdict
import plotting
from plotting import *
import json
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
    


def q_learning(ns,ch, num_episodes, discount_factor=0.99, alpha=0.25, epsilon=0.05):
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
    nsQ = defaultdict(lambda: np.zeros(ns.action_space.n))
    chQ = defaultdict(lambda: np.zeros(ch.action_space.n))
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
            #print(ns_state)
            #print(type(ns_state))
            # Take a step
            ns_action_probs = ns_policy(ns_state)
            ns_action = np.random.choice(np.arange(len(ns_action_probs)), p=ns_action_probs)
            ns_next_state, ns_reward, done, data_overflow = ns.step(ns_action)
            assert ns_action==selected_id[0]
            ns_next_state=tuple(ns_next_state.flatten())
            # Update statistics
          
            stats.episode_rewards[i_episode] += ns_reward 
            stats.episode_lengths[i_episode] = t
            stats.episode_transbag[i_episode]+=data_overflow
            # TD Update
            ns_best_next_action = np.argmax(nsQ[ns_next_state])    
            td_target = ns_reward + discount_factor * nsQ[ns_next_state][ns_best_next_action]
            td_delta = td_target - nsQ[ns_state][ns_action]
            nsQ[ns_state][ns_action] += alpha * td_delta
                
            ns_state = ns_next_state
            
            ch_state= tuple(wr.S[ns_action].flatten())
            ch_action_probs = ch_policy(ch_state)
            ch_action = np.random.choice(np.arange(len(ch_action_probs)), p=ch_action_probs)
            ch_next_state, ch_reward, _, data_overflow = ch.step(ch_action)
            ch_next_state=tuple(ch_next_state.flatten())
            stats.episode_transbag[i_episode]+=data_overflow
            # TD Update
            
            ch_best_next_action = np.argmax(chQ[ch_next_state])    
            td_target = ch_reward + discount_factor * chQ[ch_next_state][ch_best_next_action]
            td_delta = td_target - chQ[ch_state][ch_action]
            chQ[ch_state][ch_action] += alpha * td_delta
            
            
            
            if t>100:
                break



    
    return nsQ,chQ, stats

nsQ,chQ, stats = q_learning(ns,ch, 10000)
for i in range(len(stats.episode_rewards)):
    stats.episode_rewards[i]/=1

plotting.plot_episode_stats(stats)


for key in list(nsQ.keys()):
    nsQ[str(list(key))] = str(nsQ[key])
    del nsQ[key]

#print(len(nsQ))

with open("nsQ_json.json","w") as json_file:
    json_file.writelines(json.dumps(nsQ,sort_keys=True, indent=4, separators=(',', ': ')))
        
json_file.close()



for key in list(chQ.keys()):
    chQ[str(list(key))] = str(chQ[key])
    del chQ[key]
#print(len(chQ))

with open("chQ_json.json","w") as json_file:
    json_file.writelines(json.dumps(chQ,sort_keys=True, indent=4, separators=(',', ': ')))
        
json_file.close()
