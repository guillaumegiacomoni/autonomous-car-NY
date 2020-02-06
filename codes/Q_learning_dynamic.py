#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:54:41 2019

@author: guillaumegiacomoni
"""
from codes.LearningRate import LearningRate

from codes.PolicyEpsilonGreedy import PolicyEpsilonGreedy
import time
import numpy as np
from matplotlib.animation import ArtistAnimation
import pdb
import matplotlib.pyplot as plt
import os

def Q_learning(Q_, env, action_space, gamma, print_every_episode, n_episodes, epsilon, epsilon_decay, epsilon_min,
              epsilon_decay_every, lr0, lr_decay, min_lr, max_steps):
    
    """
    Calculate the Q table for the env env
    
    Args :
        Q_(np.array) : the Q table
        env(GridWorld): the grid 
        action_space(list of str): actions that are theoretically possible to do 
        print_every_episode(int): the frequency of the printing of the progress
        n_episodes(int): the number of episodes to update the Q table
        max_steps(int): the maximum number of iteration to do at each episode
        other args(float): parameter for the policy and the learning rate
        
    Returns :
        Q_ : the Q table
        policy : the definitive policy
        reward : the average reward over the episodes
        
    """
    
    policy = PolicyEpsilonGreedy(Q_, action_space, epsilon, epsilon_decay, epsilon_min,
                                  epsilon_decay_every)
    lr = LearningRate(lr0=lr0, decay=lr_decay, min_lr=min_lr, discrete=Q_._discrete,
                      actions_size=action_space, state_shape=Q_.W.shape[:-1])
    
    # Let's save the rewards of every episode so we can plot the learning curve
    rewards = [0]*n_episodes
    start = time.time()
    print("Starting training\n%i episodes" % (n_episodes))
    for episode in range(n_episodes):
        # Initialisation of the episode
        episode_reward = 0
        state = env.reset()
        for t in range(max_steps):
            action = policy(state)
            res = env.step(action)
            new_state, reward, done = res[:3]
            episode_reward += reward
            if done:  
                Q_.update(reward, state, action, lr(state, action))
                break
                
            else:
                Q_.update(reward +gamma*Q_.get_V(new_state),  state, action, lr(state, action))
                
            state = new_state 
        
        rewards[episode]+=episode_reward
                
        if (episode % print_every_episode == 0 and episode>0) or episode==n_episodes-1:
            print("Finished  episode %i of %i episodes. Mean reward: %.1f" % \
                  (episode, n_episodes, np.mean(rewards[episode-print_every_episode: episode]))\
                 + " (%.2f)" % (time.time()-start))
            
            policy._do_decay()
    policy._Q = Q_        
    return Q_, policy, rewards

def Q_learning_dynamic(Q_, env, gamma, print_every_episode, n_episodes, epsilon, epsilon_decay, epsilon_min,
              epsilon_decay_every, lr0, lr_decay, min_lr, max_steps,display = True) :
    """
    Calculate Q-table at each modification of the env env and displays the final path taken 
    
    Args :
        Q_(np.array) : the Q table
        env(GridWorld): the grid 
        print_every_episode(int): the frequency of the printing of the progress
        n_episodes(int): the number of episodes to update the Q table
        max_steps(int): the maximum number of iteration to do at each episode
        other args(float): parameter for the policy and the learning rate
        display(bool): displays the board at each modification of the env env
        
    Returns :
        env(GridWorld)
    """
    
    
    done = (env._IA == env._end_coordinates)
    dynamic_path = [env._start_coordinates]
    start_coordinates = env._start_coordinates
    t = 0
    
    while not(done):
        
        _, policy,rewards = Q_learning(Q_=env.Q, env=env, action_space= ["up", "right", "down", "left","stop"],
                                print_every_episode=print_every_episode, n_episodes=n_episodes, gamma=gamma, max_steps=max_steps,
                                epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, epsilon_decay_every=epsilon_decay_every, 
                                lr0=lr0, lr_decay=lr_decay, min_lr=min_lr)
        
        env.reset(t)
        env._IA = dynamic_path[t]
        
        state = env._IA
        
        action = policy(state, be_greedy=True)
        state, _,done, _ = env.step(action)
        
        dynamic_path.append(env._IA)
        env.next_table()
        env._start_coordinates = env._IA
        
        t += 1
        done = (env._IA == env._end_coordinates)
        
        
    env._start_coordinates = start_coordinates
    
    env.path = dynamic_path

    if display :
        env.render_path_and_V()
        for i in range(len(dynamic_path)) :
            env._IA = dynamic_path[i]
            env.reset(i)
            env.render_board(show = False,save = "{}.png".format(i))
    
    env.path = dynamic_path

    
    return env
        