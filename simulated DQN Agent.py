import torch
import random
import numpy as np
from collections import deque, Counter
import matplotlib.pyplot as plt
from flask import Flask, session, make_response
import io
import sys

# Add the path to the directory containing the DQN implementation
sys.path.append("C:/Users/Almid/PycharmProjects/Deep-Q-Network")
from Deep_Q_Network import model, choose_action, remember, replay, difficulty_levels

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"

def choose_action(state, epsilon):
    q_values = model(torch.FloatTensor(state).unsqueeze(0))
    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(1, len(difficulty_levels) + 1))  # Random action
        print(f"Exploration: Random action chosen: {action}")
    else:
        action = torch.argmax(q_values).item() + 1  # Best action based on Q-values
        print(f"Exploitation: Action chosen based on Q-values: {action}")
        print(f"Q-values for state {state}: {np.round(q_values.detach().numpy()[0], 2)}")
    return action

def simulate_agent(episodes=1000):
    state = [1]  # Starting level
    level_counts = Counter()

    epsilon = 0.5  # Start with a moderate exploration rate
    epsilon_decay = 0.99  # Faster decay of epsilon to encourage exploitation
    epsilon_min = 0.1

    total_rewards = []

    for episode in range(episodes):
        print(f"\n=== Episode {episode + 1} ===")
        print("-----------------------------------------------------")

        episode_reward = 0

        for step in range(1):  # One step per episode
            action = choose_action(state, epsilon)  # Use a dynamic epsilon for testing
            next_state = [action]

            # Update level frequency count
            level_counts[state[0]] += 1

            # Generate a mock reward: higher levels give significantly higher rewards
            if next_state[0] > state[0]:
                reward = 30 * next_state[0]  # Increased reward for advancing to higher levels
            elif next_state[0] == state[0]:
                reward = 15 * next_state[0]  # Reward for maintaining difficulty
            else:
                reward = -20 * next_state[0]  # Increased penalty for reducing difficulty

            episode_reward += reward

            # Log the result of the step
            print("Step | Current State | Action | Next State | Reward")
            print(f" {step + 1:<4}|       {state[0]:<8}|    {action:<4}|      {next_state[0]:<6}|   {reward:<6}")
            print("-----------------------------------------------------")

            # Remember and update the model
            remember(state, action - 1, reward, next_state, done=False)
            replay()

            state = next_state

        total_rewards.append(episode_reward)
        print("Replay executed. Q-network updated.")

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # Plot the frequency of difficulty levels
    levels = sorted(level_counts.keys())
    frequencies = [level_counts[level] for level in levels]

    # Bar Plot
    plt.figure()
    plt.bar(levels, frequencies, color='blue', alpha=0.7)
    plt.xlabel('Difficulty Level')
    plt.ylabel('Occurrences')
    plt.title('Time Spent on Each Difficulty Level')
    plt.tight_layout()
    plt.show()

    # Line Plot
    plt.figure()
    plt.plot(levels, frequencies, marker='o', linestyle='-', color='green')
    plt.xlabel('Difficulty Level')
    plt.ylabel('Occurrences')
    plt.title('Trend of Time Spent on Difficulty Levels')
    plt.tight_layout()
    plt.show()

    # Pie Chart
    plt.figure()
    plt.pie(frequencies, labels=[f"Level {level}" for level in levels], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Percentage of Time Spent on Each Difficulty Level')
    plt.tight_layout()
    plt.show()

@app.route('/plot.png')
def plot_png():
    rewards = session.get('rewards_per_episode', [1])  # Default to avoid empty array
    if len(rewards) == 1 and rewards[0] == 1:
        return "No rewards data found in session."

    plt.figure()
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xscale('linear')  # Lineare Skalierung der x-Achse (z. B. Episoden)
    plt.yscale('log')  # Logarithmische Skalierung der y-Achse für die Belohnungen
    plt.xlabel('Episode')
    plt.ylabel('Gesamt-Belohnung (logarithmisch)')
    plt.title('Logarithmische Darstellung der Belohnung über Episoden')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return make_response(img.read())

if __name__ == "__main__":
    simulate_agent()
