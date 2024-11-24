from flask import Flask, request, jsonify, render_template
import torch
from dqn_model import model, solve_equation, generate_algebraic_problem, knowledge_base

# Flask-App-Initialisierung
app = Flask(__name__, static_folder='static', template_folder='templates')


# Index Route
@app.route('/')
def index():
    return render_template('index.html')


# API-Route für Aufgaben-Generierung
@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    difficulty = data.get('difficulty', 1)
    equation, topic = generate_algebraic_problem(difficulty)
    solution = solve_equation(equation)

    knowledge = knowledge_base.get(topic, {})

    return jsonify({
        'equation': str(equation),
        'solution': str(solution),
        'knowledge': knowledge
    })


# API-Route für DQN-Vorhersage
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    state = data.get('state', 0)
    state_tensor = torch.tensor([[state]], dtype=torch.float32)
    prediction = model(state_tensor).tolist()[0]

    return jsonify({'predictions': prediction})


# Main
if __name__ == '__main__':
    app.run(debug=True)
