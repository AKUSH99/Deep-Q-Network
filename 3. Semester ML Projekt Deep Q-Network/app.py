from flask import Flask, render_template, request, jsonify
from sympy import Eq, symbols
from app_DQN_webless import model, generate_algebraic_problem, knowledge_base, solve_equation, plot_heatmap_live
import torch

app = Flask(__name__)

x = symbols('x')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_task():
    state = torch.randint(0, 5, (1,)).item()  # Zufälliger initialer Zustand
    equation, topic = generate_algebraic_problem(state)
    if equation is None:
        return jsonify({'error': 'Keine Aufgabe für diesen Schwierigkeitsgrad.'}), 400
    return jsonify({'equation': str(equation), 'topic': topic})

@app.route('/solve', methods=['POST'])
def solve_task():
    data = request.get_json()
    equation = eval(data.get('equation'), {"Eq": Eq, "x": x})
    topic = data.get('topic')
    solutions = solve_equation(equation)
    return jsonify({'solutions': [str(sol) for sol in solutions], 'topic': knowledge_base.get(topic, {})})

@app.route('/update_plots')
def update_plots():
    plot_heatmap_live()
    return jsonify({'status': 'Plots updated!'})

if __name__ == '__main__':
    app.run(debug=True)
