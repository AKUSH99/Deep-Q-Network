import random
from sympy import symbols, Eq, solve, latex
import matplotlib.pyplot as plt

# Definiere die Variable x
x = symbols('x')


# Funktion zur Generierung von Aufgaben auf verschiedenen Schwierigkeitsstufen
def generate_problem(level):
    if level == 1:  # Stufe 1: Einfache lineare Gleichungen
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(1, 10)
        equation = Eq(a * x + b, c)

    elif level == 2:  # Stufe 2: Leicht komplexe Gleichungen mit Klammern
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(1, 10)
        equation = Eq(a * (x + b), c)

    elif level == 3:  # Stufe 3: Mittelkomplexe Gleichungen
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(1, 10)
        d = random.randint(1, 10)
        equation = Eq(a * x + b, c * x + d)

    elif level == 4:  # Stufe 4: Schwierige Gleichungen mit mehreren Klammern
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(1, 10)
        d = random.randint(1, 10)
        equation = Eq(a * (x + b) + c * (x - d), random.randint(1, 20))

    elif level == 5:  # Stufe 5: Sehr schwierige Gleichungen (quadratische Gleichungen)
        a = random.randint(1, 5)
        b = random.randint(1, 10)
        c = random.randint(1, 10)
        equation = Eq(a * x ** 2 + b * x + c, random.randint(1, 20))

    return equation


# Funktion zum Lösen der algebraischen Aufgabe
def solve_algebra_problem(equation):
    solution = solve(equation, x)
    return solution


# Funktion zur visuellen Darstellung der Gleichung und Lösung
def plot_algebra_problem(equation, solution):
    # Mathematische Notation der Gleichung mit LaTeX
    equation_latex = latex(equation)
    solution_latex = latex(solution)

    # Plot für die Gleichung und die Lösung
    plt.figure(figsize=(6, 4))
    plt.text(0.1, 0.6, f"Aufgabe:\n${equation_latex}$", fontsize=16)
    plt.text(0.1, 0.4, f"Lösung:\n${solution_latex}$", fontsize=16)

    # Titel und Achsen ausblenden
    plt.axis('off')

    # Zeige den Plot
    plt.show()


# Generiere eine Aufgabe für eine bestimmte Stufe
level = 3  # Hier kannst du den Schwierigkeitsgrad wählen (1 bis 5)
problem = generate_problem(level)
print(f"Generierte Aufgabe: {problem}")

# Lösen der Aufgabe
solution = solve_algebra_problem(problem)
print(f"Lösung der Aufgabe: {solution}")

# Visuelle Darstellung der Aufgabe und der Lösung in mathematischer Notation
plot_algebra_problem(problem, solution)
