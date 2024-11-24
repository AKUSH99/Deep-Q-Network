import numpy as np
import random
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt

# Q-Learning Parameter
alpha = 0.1  # Lernrate
gamma = 0.9  # Discount-Faktor
epsilon = 1.0  # Start mit hoher Exploration, wird später dynamisch reduziert

# Wir haben nun 5 Schwierigkeitsstufen
difficulty_levels = [1, 2, 3, 4, 5]  # Schwierigkeitsstufen

# Q-Tabelle für fünf Schwierigkeitsgrade und drei Aktionen
q_table = np.zeros((len(difficulty_levels), 3))

x = symbols('x')

# Funktion zur Generierung linearer und quadratischer Gleichungen mit angepasster Schwierigkeit
def generate_algebraic_problem(difficulty):
    if difficulty == 1:
        a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 10)
        equation = Eq(a * x + b, c)
    elif difficulty == 2:
        a, b, c, d = random.randint(5, 10), random.randint(1, 5), random.randint(5, 10), random.randint(1, 5)
        equation = Eq(a * x + b, c * x + d)
    elif difficulty == 3:
        a, b, c = random.randint(1, 5), random.randint(1, 10), random.randint(1, 10)
        equation = Eq(a * x ** 2 + b * x, c)
    elif difficulty == 4:
        a, b, c = random.randint(5, 15), random.randint(10, 20), random.randint(5, 15)
        equation = Eq(a * x ** 2 + b * x, c)
    else:
        a, b, c, d = random.randint(10, 20), random.randint(5, 15), random.randint(10, 20), random.randint(5, 15)
        equation = Eq(a * x ** 2 + b * x + c, d * x ** 2 + b * x)

    return equation

# Funktion zur Bewertung der Erfolgswahrscheinlichkeit für algebraische Aufgaben
def evaluate_problem(equation, difficulty):
    solution = solve(equation, x)

    if solution:
        if difficulty == 1:
            success_probability = 0.9
        elif difficulty == 2:
            success_probability = 0.8
        elif difficulty == 3:
            success_probability = 0.7
        elif difficulty == 4:
            success_probability = 0.6
        else:
            success_probability = 0.5
        return random.random() < success_probability
    return False

# Dynamische Belohnungen: Anpassung an die Leistung des Lernenden
def adjust_reward(task_success, difficulty, consecutive_successes, low_level_count, last_action, action):
    reward = 0

    if task_success:
        if difficulty == 5:
            reward = 50 + (consecutive_successes * 2)
        elif difficulty == 4:
            reward = 20 + consecutive_successes
        elif difficulty == 3:
            reward = 10
        elif difficulty == 2:
            reward = 2
        else:
            reward = 1 if low_level_count < 3 else -5

        if action == 1 and difficulty == 3:
            reward -= 3
    else:
        if action == 0:
            reward = -10
        else:
            if difficulty == 5:
                reward = -5
            elif difficulty == 4:
                reward = -3
            elif difficulty == 3:
                reward = -2
            else:
                reward = -1

    if action != 1 and action == last_action and difficulty < 5:
        reward -= 2

    return reward

# Epsilon-greedy Aktion wählen
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2])
    else:
        return np.argmax(q_table[state])

# Q-Tabelle aktualisieren
def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

# Leistung des Lernenden verfolgen
performance = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
rewards_per_episode = []

# Punktzahl für den Agenten initialisieren
total_score = 0
low_level_count = 0
last_action = None
consecutive_successes = 0

# RL-Training ohne erzwungenen höheren Aufgaben
current_state = 0
for episode in range(2000):  # 2000 Episoden
    print(f"\nEpisode {episode + 1}: Schwierigkeitsgrad = {difficulty_levels[current_state]}")

    epsilon = max(0.1, epsilon * np.exp(-0.002 * episode))

    action = choose_action(current_state, epsilon)

    if action == 0:
        next_state = max(0, current_state - 1)
    elif action == 2 or consecutive_successes >= 3:
        next_state = min(4, current_state + 1)
    else:
        next_state = current_state

    if difficulty_levels[next_state] == 1:
        low_level_count += 1
    else:
        low_level_count = 0

    equation = generate_algebraic_problem(difficulty_levels[next_state])
    print(f"Generierte Aufgabe: {equation}")
    task_success = evaluate_problem(equation, difficulty_levels[next_state])

    reward = adjust_reward(task_success, next_state + 1, consecutive_successes, low_level_count, last_action, action)

    if task_success:
        performance[next_state + 1] += 1
        consecutive_successes += 1
    else:
        consecutive_successes = 0

    total_score += reward
    rewards_per_episode.append(total_score)

    update_q_table(current_state, action, reward, next_state)

    current_state = next_state
    last_action = action

    print(f"Gesamtpunktzahl nach Episode {episode + 1}: {total_score}")

# Zeige die Q-Tabelle nach dem Training in Prozent an
print("\nQ-Tabelle nach dem Training in Prozent:")
q_table_percent = (q_table / np.sum(np.abs(q_table), axis=1, keepdims=True)) * 100
print(np.round(q_table_percent, 2))

# Plot der Belohnungen über die Episoden (logarithmisch)
log_rewards_per_episode = [np.log(max(1, reward)) for reward in rewards_per_episode]

plt.plot(log_rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Logarithmische Gesamt-Belohnung')
plt.title('Logarithmische Belohnung über Episoden')
plt.show()

# Prozentuale Verteilung der gelösten Aufgaben nach Schwierigkeitsstufe
total_tasks = sum(performance.values())
percentages = {level: (count / total_tasks) * 100 for level, count in performance.items()}

# Plot der Verteilung
levels = list(percentages.keys())
values = list(percentages.values())

plt.bar(levels, values, color='skyblue')
plt.xlabel('Schwierigkeitsstufe')
plt.ylabel('Prozentuale Verteilung der gelösten Aufgaben')
plt.title('Prozentuale Verteilung der gelösten Aufgaben nach Schwierigkeitsstufe')
plt.xticks(levels, ['Stufe 1', 'Stufe 2', 'Stufe 3', 'Stufe 4', 'Stufe 5'])
plt.show()
