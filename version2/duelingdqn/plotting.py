import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
EpisodeStats = namedtuple("Stats",["episode_transbag", "episode_rewards","episode_lossbag"])

def plot_episode_stats(stats, smoothing_window=10, noshow=False,stats_comp=None):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_transbag,label="dqn")
    if stats_comp != None:
        plt.plot(stats_comp.episode_transbag,label="greedy")

    plt.xlabel("Episode")
    plt.ylabel("Episode transbag")
    plt.title("Episode transbag over Time")
    plt.legend()
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed,label="dqn")
    if stats_comp != None:
        rewards_smoothed_c = pd.Series(stats_comp.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed_c,label="greedy")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.legend()
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lossbag,label="dqn")
    if stats_comp != None:
        plt.plot(stats_comp.episode_lossbag,label="greedy")
    plt.xlabel("Episode")
    plt.ylabel("Episode lossbag")
    plt.title("Episode lossbag over Time")
    plt.legend()
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3
