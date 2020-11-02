import sys
import random
import math
from collections import defaultdict, deque

import gym
import numpy as np
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values


def epsilon_greedy(Q, state, nA, eps):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(nA))


class TemporalDifference:
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
    

    def update_Q_sarsa(self, Q, state, action, reward, next_state=None, next_action=None):
        """
        Only next state-action pair is requiered for updating the current action-value estimate
        """
        current_estimate = Q[state][action]
        next_estimate = Q[next_state][next_action] if next_state is not None else 0    
        target = reward + (self.gamma * next_estimate)
        new_value = current_estimate + (self.alpha * (target - current_estimate))
        return new_value

    def update_Q_sarsamax(self, Q, state, action, reward, next_state=None):
        """Only next state is requiered for updating the current action-value estimate"""
        current_estimate = Q[state][action]
        next_estimate = np.max(Q[next_state]) if next_state is not None else 0
        target = reward + (self.gamma * next_estimate)
        new_value = current_estimate + (self.alpha * (target - current_estimate))
        return new_value

    
    def learn(self, epochs, update='sarsa', plot_every=100):
        nA = self.env.action_space.n
        Q = defaultdict(lambda: np.zeros(nA))
        
        tmp_scores = deque(maxlen=plot_every)
        avg_scores = deque(maxlen=epochs)
        
        for epoch in range(1, epochs+1):
            if epoch % 100 == 0:
                print("\rEpisode {}/{}".format(epoch, epochs), end="")
                sys.stdout.flush()   
            
            score = 0
            current_state = self.env.reset()
            
            eps = 1.0 / epoch
            current_action = epsilon_greedy(Q, current_state, nA, eps)
            
            while True:
                next_state, reward, done, info = self.env.step(current_action)
                score += reward
                
                if not done:
                    next_action = epsilon_greedy(Q, next_state, nA, eps)

                    if update == 'sarsa':
                        new_val = self.update_Q_sarsa(Q, current_state, current_action, reward, next_state, next_action)
                    elif update == 'q-learning':
                        new_val = self.update_Q_sarsamax(Q, current_state, current_action, reward, next_state)
                    
                    
                    Q[current_state][current_action] = new_val
                    current_state = next_state
                    current_action = next_action

                if done:
                    Q[current_state][current_action] = self.update_Q_sarsa(Q, current_state, current_action, reward)
                    tmp_scores.append(score)
                    break

            if (epoch % plot_every == 0):
                avg_scores.append(np.mean(tmp_scores))

        plt.plot(np.linspace(0,epochs,len(avg_scores),endpoint=False), np.asarray(avg_scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.show()
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))    
        return Q


if __name__ == "__main__":

    env = gym.make('CliffWalking-v0')
    temp_diff = TemporalDifference(env, alpha=.01, gamma=.9)
    Q = temp_diff.learn(epochs=8000, update='q-learning', plot_every=100)

    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    policy = np.array([np.argmax(Q[state]) if state in Q else -1 for state in np.arange(48)]).reshape(4,12)
    print(policy)

    # plot the estimated optimal state-value function
    V = ([np.max(Q[state]) if state in Q else 0 for state in np.arange(48)])
    plot_values(V)