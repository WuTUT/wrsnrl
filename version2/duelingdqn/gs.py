#greedy select
import wrsn2
import gym
from wrsn2 import *
import sys
if "../" not in sys.path:
  sys.path.append("../")
import matplotlib
import plotting
from plotting import *
import itertools
import os
import random
import ddqn
from ddqn import Estimator
import tensorflow as tf
def greedyselect_known(state):
    action=0
    B,D,H=state[:,0],state[:,1],state[:,2]
    
    sensor_id=np.argmax(D)
    if funPh(H[sensor_id])<=B[sensor_id]:
        action=sensor_id*2+1
    else:
        action=sensor_id*2
    
    return action




if __name__ == "__main__":
    env=wrsn(sensor_node)
    num_episodes=500
    stats = plotting.EpisodeStats(
        episode_transbag=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_lossbag=np.zeros(num_episodes))
    
    
    # tf.reset_default_graph()

    # # Where we save our checkpoints and graphs
    # experiment_dir = os.path.abspath("./experiments/wrsn2")
    # q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    
    
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    #     checkpoint_path = os.path.join(checkpoint_dir, "model")
    #     saver = tf.train.Saver()
    #     latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    #     if latest_checkpoint:
    #         print("Loading model checkpoint {}...\n".format(latest_checkpoint))
    #         saver.restore(sess, latest_checkpoint)
    #     total_t = sess.run(tf.contrib.framework.get_global_step())

    #     policy =ddqn.make_epsilon_greedy_policy(q_estimator,env.action_space.n)
    #     for i_episode in range(num_episodes):
    #         print("\rEpisode {}/{}".format(i_episode + 1, num_episodes), end="")
    #         sys.stdout.flush()

    #         state = env.reset_test()
    #         for t in itertools.count():
    #             action_probs = policy(sess, state, 0.0)
    #             action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
    #             next_state, reward, done, data_overflow,data_trans = env.step(action)
    #             stats.episode_transbag[i_episode] += data_trans
    #             stats.episode_rewards[i_episode] += reward
    #             stats.episode_lossbag[i_episode] += data_overflow
    #             if t>=4000:
    #                 break
    #             state = next_state
    #             total_t += 1

    stats_c = plotting.EpisodeStats(
        episode_transbag=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_lossbag=np.zeros(num_episodes))      
    _=env.reset_test()    
    for i_episode in range(num_episodes):
        

        _ = env.reset_test()
        for t in itertools.count():
            best_action=greedyselect_known(env.S)
            _, reward, _, data_overflow,data_trans=env.step(best_action)
            stats_c.episode_transbag[i_episode] += data_trans
            stats_c.episode_rewards[i_episode] += reward
            stats_c.episode_lossbag[i_episode] += data_overflow
            if t>=1000:
                break
        print("\rEpisode {} / {} , lossbag {}  D {}  ".format(i_episode + 1, num_episodes,stats_c.episode_lossbag[i_episode],np.sum(env.S[:,1])), end="")
        sys.stdout.flush()
    print("\n")
    print(np.mean(stats_c.episode_lossbag))
    plotting.plot_episode_stats(stats,stats_comp=stats_c)

            