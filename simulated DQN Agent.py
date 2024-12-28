import torch
import random
import numpy as np
from collections import deque, Counter
import sys
import matplotlib.pyplot as plt

# Add the path to the directory containing the DQN implementation
sys.path.append("C:/Users/Almid/PycharmProjects/Deep-Q-Network")  # Update with the correct path if needed
from Deep_Q_Network import model, choose_action, remember, replay, difficulty_levels  # Import based on the correct file name

# Assuming the DQN and supporting code has been imported

def simulate_agent(episodes=2000):
    state = [1]  # Starting level
    level_counts = Counter()

    epsilon = 0.5  # Start with a moderate exploration rate
    epsilon_decay = 0.99  # Faster decay of epsilon to encourage exploitation
    epsilon_min = 0.1

    for episode in range(episodes):
        print(f"\n=== Episode {episode + 1} ===")
        print("Step | Current State | Action | Next State | Reward")
        print("-----------------------------------------------------")

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

            # Log the result of the step
            print(f"  {step + 1}  |       {state[0]}       |   {action}    |     {next_state[0]}      |   {reward}")

            # Remember and update the model
            remember(state, action - 1, reward, next_state, done=False)
            replay()

            state = next_state

        # Display Q-values for the last state of the episode
        q_values = model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
        q_values_rounded = np.round(q_values[0], 2)
        print("\nQ-Values for Current State:")
        print(q_values_rounded)
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

    # Improved Neural Network Architecture Visualization
    def visualize_architecture():
        layers = ["Input", "Hidden Layer 1", "Hidden Layer 2", "Output"]
        sizes = [1, 24, 24, len(difficulty_levels)]

        fig, ax = plt.subplots(figsize=(12, 8))
        x_coords = range(len(layers))
        max_frequency = max(level_counts.values()) if level_counts else 1

        for i, size in enumerate(sizes):
            y_coords = np.linspace(0, 1, size)
            ax.scatter([x_coords[i]] * size, y_coords, s=100, label=f"{layers[i]} ({size})", zorder=3, color=['blue', 'orange', 'green', 'red'][i])

        for i in range(len(sizes) - 1):
            for y1 in np.linspace(0, 1, sizes[i]):
                for y2 in np.linspace(0, 1, sizes[i + 1]):
                    level_index = i + 1
                    weight = level_counts[level_index] if level_index in level_counts else 0
                    normalized_weight = weight / max_frequency
                    ax.plot(
                        [x_coords[i], x_coords[i + 1]],
                        [y1, y2],
                        color=plt.cm.viridis(normalized_weight),
                        alpha=0.6 + 0.4 * normalized_weight,
                        linewidth=0.5 + 2 * normalized_weight,
                        zorder=1
                    )

        # Annotate difficulty levels on the output layer
        for i, label in enumerate(difficulty_levels):
            ax.annotate(f"Level {label}\n{level_counts.get(label, 0)} occurrences",
                        (x_coords[-1] + 0.1, np.linspace(0, 1, sizes[-1])[i]),
                        fontsize=10, fontweight='bold', color='red')

        ax.set_xticks(x_coords)
        ax.set_xticklabels(layers, fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.set_title("Neural Network Architecture with Level Frequencies", fontsize=16, fontweight='bold')
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()
        plt.show()

    visualize_architecture()

if __name__ == "__main__":
    simulate_agent()
