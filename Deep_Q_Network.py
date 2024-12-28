from flask import Flask, render_template, request, session, make_response
import matplotlib.pyplot as plt
import io
import numpy as np
import random
from sympy import symbols, Eq, solve, latex
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from decimal import Decimal
import cmath

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Q-Learning Parameter
alpha = 0.5
gamma = 0.99
epsilon = 0.1

difficulty_levels = [1, 2, 3, 4, 5]
x = symbols('x')

# DQN Parameter
state_size = 1
action_size = len(difficulty_levels)
batch_size = 64

# Experience Replay
memory = deque(maxlen=2000)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()


def generate_algebraic_problem(difficulty):
    if difficulty == 1:  # Simple linear equation with one solution
        a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 10)
        equation = Eq(a * x + b, c)
    elif difficulty == 2:  # Linear equation with one solution (ax + b = cx + d)
        a, b, c, d = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
        while a == c:  # Ensure non-zero slope difference
            a, c = random.randint(1, 10), random.randint(1, 10)
        equation = Eq(a * x + b, c * x + d)
    elif difficulty == 3:  # Quadratic equation with two real solutions
        while True:
            a = random.randint(1, 10)
            b = random.randint(1, 20)
            c = random.randint(1, 20)
            discriminant = b ** 2 - 4 * a * c
            if discriminant >= 0:  # Ensure two real roots
                break
        equation = Eq(a * x ** 2 + b * x, c)
    elif difficulty == 4:  # More challenging quadratic equations
        while True:
            a = random.randint(1, 15)
            b = random.randint(5, 25)
            c = random.randint(5, 20)
            discriminant = b ** 2 - 4 * a * c
            if discriminant >= 0:  # Ensure two real roots
                break
        equation = Eq(a * x ** 2 + b * x + c, 0)
    else:  # Difficulty 5: Complex quadratic equations with terms on both sides
        while True:
            a, b, c = random.randint(1, 10), random.randint(1, 15), random.randint(1, 10)
            d, e, f = random.randint(1, 10), random.randint(1, 15), random.randint(1, 10)
            discriminant_left = b ** 2 - 4 * a * c
            discriminant_right = e ** 2 - 4 * d * f
            if discriminant_left >= 0 and discriminant_right >= 0:  # Ensure both sides have real solutions
                break
        equation = Eq(a * x ** 2 + b * x + c, d * x ** 2 + e * x + f)
    return equation


def choose_action(state, epsilon):
    q_values = model(torch.FloatTensor(state).unsqueeze(0))
    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(1, action_size + 1))
        print(f"Exploration: Random action chosen: {action}")
    else:
        action = torch.argmax(q_values).item() + 1
        print(f"Exploitation: Action chosen based on Q-values: {action}")
    print(f"Q-values for state {state}: {q_values.detach().numpy()}")
    return action


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * torch.max(model(torch.FloatTensor(next_state).unsqueeze(0))).item()
        target_f = model(torch.FloatTensor(state).unsqueeze(0))
        target_f[0][action] = target
        optimizer.zero_grad()
        loss = loss_fn(target_f, model(torch.FloatTensor(state).unsqueeze(0)))
        loss.backward()
        optimizer.step()
    print("Replay executed. Q-network updated.")


@app.route('/')
def index():
    # Reset session data only when leaving the application
    if 'total_score' not in session or 'current_state' not in session or not session['current_state'] or \
            session['current_state'][0] < 1 or session['current_state'][0] > len(difficulty_levels):
        session['current_state'] = [1];
        session['total_score'] = 0;
        session['rewards_per_episode'] = []
        session['total_score'] = 0
        session['rewards_per_episode'] = []

    equation = generate_algebraic_problem(difficulty_levels[session['current_state'][0] - 1])
    solution = solve(equation, x)
    session['current_equation'] = latex(equation)

    # Update correct_solution to include only real solutions
    session['correct_solution'] = [str(round(float(s), 3)) for s in solution if s.is_real]
    return render_template('index.html', equation=session['current_equation'],
                           current_level=session['current_state'][0], total_score=session['total_score'],
                           current_state_feedback=f'Stufe: {session["current_state"][0]}')


@app.route('/solve', methods=['POST'])
def solve_problem():
    user_solution_1 = request.form.get('solution_1', '').strip()
    user_solution_2 = request.form.get('solution_2', '').strip()
    correct_solution = session.get('correct_solution', [])
    state = session.get('current_state', [1])
    if state[0] < 1 or state[0] > len(difficulty_levels):
        state = [1]
        session['current_state'] = state
    current_state = state[0]

    is_correct = False
    try:
        user_solution_1 = float(user_solution_1) if user_solution_1 else None
        user_solution_2 = float(user_solution_2) if user_solution_2 else None
        user_solutions = [user_solution_1, user_solution_2]

        # Check for correct answers with tolerance
        for sol in correct_solution:
            if any(abs(user_sol - float(sol)) < 0.01 for user_sol in user_solutions if user_sol is not None):
                is_correct = True
                break
    except ValueError:
        is_correct = False

    reward = (10 * current_state) if is_correct else (-5 * current_state)
    session['total_score'] += reward
    session['rewards_per_episode'].append(session['total_score'])

    # DQN entscheidet die nächste Schwierigkeit basierend auf der Aktion
    action = max(0, min(choose_action(state, epsilon), len(difficulty_levels) - 1))
    next_state = [action + 1]
    remember(state, action, reward, next_state, is_correct)
    replay()

    previous_state = current_state  # Keep track of the previous level
    session['current_state'] = next_state

    # Feedback zur Schwierigkeitsstufenänderung
    if next_state[0] > previous_state:
        level_feedback = f'Die Schwierigkeit wurde auf Level {next_state[0]} ERHÖHT.'
    elif next_state[0] < previous_state:
        level_feedback = f'Die Schwierigkeit wurde auf Level {next_state[0]} VERRINGERT.'
    else:
        level_feedback = f'Die Schwierigkeit blieb auf Level {current_state}.'

    print(f"User's solutions: {user_solutions}")
    print(f"Correct solutions: {correct_solution}")
    print(f"Reward: {reward}, Total Score: {session['total_score']}, Level Feedback: {level_feedback}")

    return render_template('result.html', current_level=current_state, next_level=next_state[0],
                           equation=session.get('current_equation'),
                           user_solution_1=user_solution_1, user_solution_2=user_solution_2,
                           correct_solution=correct_solution, feedback="Richtig!" if is_correct else "Falsch!",
                           total_score=session['total_score'], level_feedback=level_feedback)


@app.route('/plot.png')
def plot_png():
    rewards = session.get('rewards_per_episode', [])
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Gesamt-Belohnung')
    plt.title('Belohnung über Episoden')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return make_response(img.read())


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
