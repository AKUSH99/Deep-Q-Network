import numpy as np
import random
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

# Q-Learning Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.7
difficulty_levels = [1, 2, 3, 4, 5]
x = symbols('x')

# Wissensbasis
knowledge_base = {
    "linear_equation": {
        "concept": "Eine lineare Gleichung hat die Form ax + b = c.",
        "steps": [
            "Subtrahiere b auf beiden Seiten.",
            "Teile durch a, um x zu isolieren."
        ],
        "example": "2x + 3 = 7 -> 2x = 4 -> x = 2"
    },
    "quadratic_equation": {
        "concept": "Quadratische Gleichungen haben die Form ax^2 + bx + c = 0.",
        "steps": [
            "Verwende die Mitternachtsformel: x = (-b ± sqrt(b^2 - 4ac)) / 2a.",
            "Bestimme die Diskriminante (b^2 - 4ac)."
        ],
        "example": "x^2 - 5x + 6 = 0 -> x1 = 2, x2 = 3"
    }
}

# DQN Model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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
        return Eq(a * x + b, c), "linear_equation"
    elif difficulty == 2:
        a, b, c, d = random.randint(1, 5), random.randint(1, 5), random.randint(1, 10), random.randint(1, 5)
        return Eq(a * x + b, c * x + d), "linear_equation"
    elif difficulty == 3:
        a, b, c = random.randint(1, 5), random.randint(1, 10), random.randint(1, 10)
        return Eq(a * x ** 2 + b * x, c), "quadratic_equation"
    elif difficulty == 4:
        a, b, c = random.randint(5, 10), random.randint(10, 20), random.randint(5, 10)
        return Eq(a * x ** 2 + b * x, c), "quadratic_equation"
    elif difficulty == 5:
        a, b, c, d = random.randint(5, 15), random.randint(1, 10), random.randint(5, 10), random.randint(1, 10)
        return Eq(a * x ** 2 + b * x + c, d * x), "quadratic_equation"
    else:
        return None, None

def display_knowledge(topic):
    if topic in knowledge_base:
        print("\nTheoretisches Wissen zur Aufgabe:")
        print(f"Konzept: {knowledge_base[topic]['concept']}")
        print("Schritte zur Lösung:")
        for step in knowledge_base[topic]['steps']:
            print(f"  - {step}")
        print(f"Beispiel: {knowledge_base[topic]['example']}")

def plot_heatmap(model):
    with torch.no_grad():
        sample_input = torch.tensor([[0.0]], dtype=torch.float32)
        output = model(sample_input).numpy()
        heatmap_data = output.reshape(1, -1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=['Stay', 'Lower', 'Raise'], yticklabels=['Input'])
    plt.title('DQN Output Heatmap')
    plt.xlabel('Actions')
    plt.ylabel('Input State')
    plt.show()

# Main Loop
rewards = []
current_state = 0
state_counts = [0] * len(difficulty_levels)

for episode in range(500):
    equation, topic = generate_algebraic_problem(difficulty_levels[current_state])
    if equation is None:
        print(f"Episode {episode}: Keine Aufgabe für Schwierigkeit {difficulty_levels[current_state]}")
        continue

    solution = solve(equation, x)
    task_success = bool(solution)
    action = choose_action(current_state, epsilon)
    next_state = max(0, min(4, current_state + (1 if action == 2 else -1 if action == 0 else 0)))

    reward = difficulty_levels[next_state] * 10
    remember(current_state, action, reward, next_state, not task_success)

    rewards.append(reward)
    state_counts[next_state] += 1
    current_state = next_state
    replay()

    print(f"\nEpisode {episode}, State: {current_state}, Equation: {equation}, Solution: {solution}")
    display_knowledge(topic)

    if episode % 50 == 0:
        plot_heatmap(model)
