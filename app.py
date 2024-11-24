import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, make_response
import io
from sympy import symbols, Eq, solve, latex

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# DQN Parameter
alpha = 0.001  # Lernrate
gamma = 0.9  # Discount-Faktor
epsilon = 1.0  # Anfangs-Explorationsrate
epsilon_decay = 0.995  # Reduzierung der Explorationsrate pro Episode
epsilon_min = 0.01
memory = deque(maxlen=2000)  # Replay Memory
batch_size = 32  # Batch-Größe für Training

difficulty_levels = [1, 2, 3, 4, 5]
x = symbols('x')


# Neuronales Netzwerk für das DQN
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=len(difficulty_levels), activation="relu"))
    model.add(layers.Dense(24, activation="relu"))
    model.add(layers.Dense(3, activation="linear"))  # Drei Aktionen: erhöhen, bleiben, senken
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
    return model

dqn_model = build_model()


# Funktion zur Generierung algebraischer Aufgaben
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


# Aufgabe bewerten
def evaluate_problem(equation, difficulty):
    solution = solve(equation, x)
    success_probability = 0.9 - 0.1 * (difficulty - 1)
    return random.random() < success_probability, solution


# Replay Memory speichern
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# DQN trainieren
def train_dqn():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            target += gamma * np.amax(dqn_model.predict(np.array([next_state]))[0])

        target_f = dqn_model.predict(np.array([state]))
        target_f[0][action] = target
        dqn_model.fit(np.array([state]), target_f, epochs=1, verbose=0)


# Index-Routen für die Aufgabendarstellung und das DQN-Training
@app.route('/')
def index():
    current_state = session.get('current_state', 0)
    equation = generate_algebraic_problem(difficulty_levels[current_state])
    task_success, solution = evaluate_problem(equation, difficulty_levels[current_state])

    session['current_equation'] = latex(equation)
    session['correct_solution'] = str(solution[0].evalf()) if solution else "Keine Lösung"
    session['total_score'] = session.get('total_score', 0)
    session['rewards_per_episode'] = session.get('rewards_per_episode', [])

    return render_template('index.html', equation=session['current_equation'])


@app.route('/solve', methods=['POST'])
def solve_problem():
    user_solution = request.form['solution']
    correct_solution = session.get('correct_solution', "Keine Lösung")
    state = np.zeros(len(difficulty_levels))
    state[session.get('current_state', 0)] = 1

    # Vergleich der Lösungen als Dezimalzahl
    try:
        user_solution_decimal = float(user_solution)
        correct_solution_decimal = float(correct_solution)
        is_correct = abs(user_solution_decimal - correct_solution_decimal) < 0.01
    except:
        is_correct = False

    reward = 10 if is_correct else -5
    session['total_score'] += reward
    session['rewards_per_episode'].append(session['total_score'])

    # DQN: Aktion und Update des Modells
    done = True  # Jede Episode endet nach einer Aufgabe
    next_state = state  # Ersetzen, falls eine neue Zustandübergang benötigt wird
    action = np.argmax(dqn_model.predict(state.reshape(1, -1))[0])

    store_experience(state, action, reward, next_state, done)
    train_dqn()

    feedback = "Richtig!" if is_correct else "Falsch!"
    return render_template('result.html', equation=session.get('current_equation'), user_solution=user_solution,
                           correct_solution=correct_solution, feedback=feedback,
                           total_score=session['total_score'])


if __name__ == "__main__":
    app.run(debug=True)
