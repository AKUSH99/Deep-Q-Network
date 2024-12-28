import torch
import random
import numpy as np
from collections import deque
import sys

# Add the path to the directory containing the DQN implementation
sys.path.append("C:/Users/Almid/PycharmProjects/Deep-Q-Network")  # Update with the correct path if needed
from Deep_Q_Network import model, choose_action, remember, replay, difficulty_levels  # Import based on the correct file name

# Assuming the DQN and supporting code has been imported

def simulate_agent(episodes=200):
    state = [1]  # Starting level
    rewards = []

    for episode in range(episodes):
        total_reward = 0
        print(f"\nEpisode {episode + 1}:")

        for step in range(10):  # Simulate a fixed number of steps per episode
            action = choose_action(state, epsilon=0.2)  # Use a lower epsilon for testing
            next_state = [action]

            # Generate a mock reward: higher levels give higher rewards
            if next_state[0] > state[0]:
                reward = 10 * next_state[0]  # Reward for increasing difficulty
            elif next_state[0] == state[0]:
                reward = 5 * next_state[0]  # Reward for maintaining difficulty
            else:
                reward = -5 * next_state[0]  # Penalty for reducing difficulty

            total_reward += reward

            # Log the result of the step
            print(f"Step {step + 1}: State {state[0]} -> Action {action} -> Next State {next_state[0]} | Reward: {reward}")

            # Remember and update the model
            remember(state, action - 1, reward, next_state, done=False)
            replay()

            state = next_state

        rewards.append(total_reward)
        print(f"Total Reward for Episode {episode + 1}: {total_reward}")

    # Plot the rewards over episodes
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance over Episodes')
    plt.show()

if __name__ == "__main__":
    simulate_agent()
