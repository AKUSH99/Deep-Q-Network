import numpy as np
import random
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

# Q-Learning Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.7
difficulty_levels = [1, 2, 3, 4, 5]
x = symbols('x')


# DQN Model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 4)  # Input -> Hidden Layer 1
        self.fc2 = nn.Linear(4, 3)  # Hidden Layer 1 -> Output Layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize model, optimizer, and loss function
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Replay Memory
memory = []
batch_size = 32


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
    if len(memory) > 5000:
        memory.pop(0)


def replay():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor([next_state], dtype=torch.float32).unsqueeze(0)

        target = reward
        if not done:
            target = reward + gamma * torch.max(model(next_state_tensor)).item()

        output = model(state_tensor)
        target_f = output.clone().detach()
        target_f[0, action] = target  # Target for chosen action

        optimizer.zero_grad()
        loss = loss_fn(output, target_f)
        loss.backward()
        optimizer.step()


def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2])  # Explore
    else:
        state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0)
        return torch.argmax(model(state_tensor)).item()  # Exploit


def generate_algebraic_problem(difficulty):
    if difficulty == 1:
        a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 10)
        return Eq(a * x + b, c)
    elif difficulty == 2:
        a, b, c, d = random.randint(5, 10), random.randint(1, 5), random.randint(5, 10), random.randint(1, 5)
        return Eq(a * x + b, c * x + d)
    elif difficulty == 3:
        a, b, c = random.randint(1, 5), random.randint(1, 10), random.randint(1, 10)
        return Eq(a * x ** 2 + b * x, c)
    elif difficulty == 4:
        a, b, c = random.randint(5, 15), random.randint(10, 20), random.randint(5, 15)
        return Eq(a * x ** 2 + b * x, c)
    else:
        a, b, c, d = random.randint(10, 20), random.randint(5, 15), random.randint(10, 20), random.randint(5, 15)
        return Eq(a * x ** 2 + b * x + c, d * x ** 2 + b * x)


def adjust_reward(task_success, difficulty, consecutive_successes, low_level_count, action):
    base_reward = difficulty * 10
    if task_success:
        streak_bonus = consecutive_successes * difficulty * 2
        reward = base_reward + streak_bonus
    else:
        penalty = base_reward * (1 + low_level_count / 3)
        reward = -penalty

    if action == 1 and low_level_count >= 3:  # Staying too long
        reward -= 5

    return reward


# Visualize Neural Network
def draw_neural_network(model):
    G = nx.DiGraph()

    # Define nodes
    input_nodes = ['Input']
    hidden_nodes = [f'Hidden {i}' for i in range(1, 5)]
    output_nodes = ['Lower Difficulty', 'Stay', 'Increase Difficulty']

    # Input -> Hidden
    for i in range(len(input_nodes)):
        for j in range(len(hidden_nodes)):
            weight = model.fc1.weight[j, i].item()
            G.add_edge(input_nodes[i], hidden_nodes[j], weight=round(weight, 2))

    # Hidden -> Output
    for i in range(len(hidden_nodes)):
        for j in range(len(output_nodes)):
            weight = model.fc2.weight[j, i].item()
            G.add_edge(hidden_nodes[i], output_nodes[j], weight=round(weight, 2))

    # Custom layout for better visibility
    pos = nx.multipartite_layout(G, subset_key=lambda n: 0 if n in input_nodes else (1 if n in hidden_nodes else 2))

    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Neural Network Visualization")
    plt.show()


# Main Loop
rewards = []
current_state = 0
consecutive_successes = 0
low_level_count = 0
state_counts = [0] * len(difficulty_levels)

for episode in range(50):
    equation = generate_algebraic_problem(difficulty_levels[current_state])
    solution = solve(equation, x)
    task_success = bool(solution)
    action = choose_action(current_state, epsilon)
    next_state = max(0, min(4, current_state + (1 if action == 2 else -1 if action == 0 else 0)))

    reward = adjust_reward(task_success, difficulty_levels[next_state], consecutive_successes, low_level_count, action)
    remember(current_state, action, reward, next_state, not task_success)

    if task_success:
        consecutive_successes += 1
    else:
        consecutive_successes = 0

    low_level_count = low_level_count + 1 if next_state == 0 else 0
    rewards.append(reward)
    state_counts[next_state] += 1
    current_state = next_state
    replay()

    if episode % 10 == 0:
        print(f"Episode {episode}, Current State: {current_state}, Reward: {sum(rewards)}")
        draw_neural_network(model)

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(rewards))
plt.yscale('log')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward (log scale)')
plt.title('Cumulative Rewards over Time')
plt.show()

plt.bar(difficulty_levels, state_counts)
plt.xlabel('Difficulty Level')
plt.ylabel('State Visits')
plt.title('State Distribution')
plt.show()
