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
alpha = 0.3
gamma = 0.9
epsilon = 0.7

difficulty_levels = [1, 2, 3, 4, 5]
x = symbols('x')

# DQN Parameter
state_size = 1
action_size = len(difficulty_levels)
batch_size = 32

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
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


def generate_algebraic_problem(difficulty):
    if difficulty == 1:
        a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 10)
        equation = Eq(a * x + b, c)
    elif difficulty == 2:
        a, b, c, d = random.randint(5, 10), random.randint(1, 5), random.randint(5, 10), random.randint(1, 5)
        equation = Eq(a * x + b, c * x + d)
    elif difficulty == 3:
        a, b, c = random.randint(1, 5), random.randint(1, 10), random.randint(1, 10)
        # Ensure b is not zero to avoid equations without linear term
        b = random.randint(1, 10)
        equation = Eq(a * x ** 2 + b * x, c)
    elif difficulty == 4:
        a, b, c = random.randint(5, 15), random.randint(1, 20), random.randint(5, 15)
        # Ensure b is not zero to avoid equations without linear term
        b = random.randint(1, 20)
        equation = Eq(a * x ** 2 + b * x, c)
    else:
        a, b, c, d = random.randint(10, 20), random.randint(1, 15), random.randint(10, 20), random.randint(5, 15)
        # Ensure b is not zero to avoid equations without linear term
        b = random.randint(1, 15)
        equation = Eq(a * x ** 2 + b * x + c, d * x ** 2 + random.randint(1, 15) * x)

    return equation


def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(1, action_size + 1))
    else:
        with torch.no_grad():
            return torch.argmax(model(torch.FloatTensor(state).unsqueeze(0))).item()


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


@app.route('/')
def index():
    # Reset session data only when leaving the application
    if 'total_score' not in session or 'current_state' not in session or not session['current_state'] or session['current_state'][0] < 1 or session['current_state'][0] > len(difficulty_levels):
        session['current_state'] = [1]; session['total_score'] = 0; session['rewards_per_episode'] = []
        session['total_score'] = 0
        session['rewards_per_episode'] = []

    equation = generate_algebraic_problem(difficulty_levels[session['current_state'][0] - 1])
    solution = solve(equation, x)
    session['current_equation'] = latex(equation)
    session['correct_solution'] = [str(s.evalf()) for s in solution]
    return render_template('index.html', equation=session['current_equation'], current_level=session['current_state'][0], total_score=session['total_score'], current_state_feedback=f'Stufe: {session["current_state"][0]}')


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
        # Handling user input for complex numbers with two decimal places followed by 'i'
        if user_solution_1:
            user_solution_1 = user_solution_1.replace('i', 'j').replace('I', 'j')
            user_solution_complex_1 = complex(user_solution_1)
        else:
            user_solution_complex_1 = None

        if user_solution_2:
            user_solution_2 = user_solution_2.replace('i', 'j').replace('I', 'j')
            user_solution_complex_2 = complex(user_solution_2)
        else:
            user_solution_complex_2 = None

        for sol in correct_solution:
            correct_sol_complex = complex(sol.replace('*I', 'j').replace('I', 'j'))
            # Adjust format for comparison, allow small margin for complex numbers
            if (user_solution_complex_1 is not None and abs(user_solution_complex_1 - correct_sol_complex) < 0.01) or \
               (user_solution_complex_2 is not None and abs(user_solution_complex_2 - correct_sol_complex) < 0.01):
                is_correct = True
                break
    except ValueError:
        is_correct = False

    reward = 10 if is_correct else -5
    session['total_score'] += reward
    session['rewards_per_episode'].append(session['total_score'])

    # DQN entscheidet die nächste Schwierigkeit basierend auf der Aktion
    action = max(0, min(choose_action(state, epsilon), len(difficulty_levels) - 1))
    next_state = [action + 1]
    remember(state, action, reward, next_state, is_correct)
    replay()

    session['current_state'] = next_state

    # Nachricht zur Stufenänderung
    if next_state[0] > current_state:
        stufen_nachricht = f'Sie wurden auf Stufe {next_state[0]} hochgestuft'
    elif next_state[0] < current_state:
        stufen_nachricht = f'Sie wurden auf Stufe {next_state[0]} abgestuft'
    else:
        stufen_nachricht = f'Sie bleiben auf Stufe {current_state}'

    # Logging für Konsole
    print(f"Aktueller Zustand: {current_state}")
    print(f"Q-Werte: {model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()}")
    print(f"Gewählte Aktion: {action}")
    print(f"Belohnung: {reward}, Gesamtpunktzahl: {session['total_score']}")

    return render_template('result.html', current_level=current_state, next_level=next_state[0], next_action_feedback=stufen_nachricht, current_state_feedback=f'Stufe: {current_state}',
                           equation=session.get('current_equation'),
                           user_solution_1=user_solution_1,
                           user_solution_2=user_solution_2,
                           correct_solution=correct_solution,
                           feedback="Richtig!" if is_correct else "Falsch!", current_level_feedback=f'Aktueller Schwierigkeitsgrad: Stufe {next_state[0]}',
                           total_score=session['total_score'])


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


@app.route('/q_values_plot.png')
def q_values_plot():
    state = session.get('current_state', [0])
    q_values = model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]

    plt.figure()
    plt.bar([f'Aktion {i}' for i in range(action_size)], q_values, color='skyblue')
    plt.xlabel('Aktionen')
    plt.ylabel('Q-Werte')
    plt.title('Q-Werte für aktuelle Aktionen')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return make_response(img.read())


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
