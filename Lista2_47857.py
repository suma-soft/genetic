import pandas as pd
import numpy as np

# Wczytanie danych z pliku zbior_danych_ag.csv
data = pd.read_csv("zbior_danych_ag.csv", sep=';')

def parse_array(array_string):
    array_string = array_string.strip("[]")
    array_string = array_string.replace("  ", " ")
    elements = [int(x) for x in array_string.split()]
    return np.array(elements)

data['Ciezar'] = data['Ciezar'].apply(parse_array)
data['Ceny'] = data['Ceny'].apply(parse_array)

weights = data['Ciezar'][0]
values = data['Ceny'][0]
capacity = data['Pojemnosc'][0]

# Funkcja logowania
log_file = "logs.txt"
def log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Funkcja oceny wartości chromosomu (fitness)
def fitness(chromosome, values, weights, capacity):
    total_value = np.sum(chromosome * values)  # Suma wartości wybranych przedmiotów
    total_weight = np.sum(chromosome * weights)  # Suma wag wybranych przedmiotów
    
    if total_weight > capacity:
        # Kara za przekroczenie pojemności plecaka
        penalty = (total_weight - capacity) / capacity  # Kara proporcjonalna do przekroczenia
        return total_value * (1 - penalty)  # Obniżenie wartości rozwiązania
    else:
        return total_value  # Wartość, jeśli waga nie przekracza pojemności

# Funkcja generująca losową populację
def generate_initial_population(num_items, population_size):
    return np.random.randint(2, size=(population_size, num_items))

# Selekcja turniejowa
def tournament_selection(population, fitness_scores, tournament_size=3):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
    return population[best_index]

# Single-point crossover
def single_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    log(f"Single-point krzyżowanie w punkcie {crossover_point}")
    return child1, child2

# Two-point crossover
def two_point_crossover(parent1, parent2):
    point1 = np.random.randint(1, len(parent1) - 2)
    point2 = np.random.randint(point1 + 1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    log(f"Two-point krzyżowanie pomiędzy punktami {point1} i {point2}")
    return child1, child2

# Mutacja
def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# Generacja nowej populacji
def generate_next_generation(population, fitness_scores, mutation_rate, operator_choice):
    new_population = []
    population_size = len(population)
    for _ in range(population_size // 2):
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)

        # Wybór operatora krzyżowania
        if operator_choice == 0:
            child1, child2 = single_point_crossover(parent1, parent2)
            operator_used = "Single-point"
        else:
            child1, child2 = two_point_crossover(parent1, parent2)
            operator_used = "Two-point"

        # Mutacja
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)

        # Logowanie szczegółów
        log(f"Krzyżowanie ({operator_used}):")
        log(f"Parent1: {parent1}, Parent2: {parent2}")
        log(f"Child1: {child1}, Child2: {child2}")

        new_population.append(child1)
        new_population.append(child2)

    return np.array(new_population)

# Algorytm genetyczny
def genetic_algorithm(values, weights, capacity, num_items, population_size, generations, mutation_rate, operator_choice):
    population = generate_initial_population(num_items, population_size)
    for generation in range(generations):
        fitness_scores = [fitness(ind, values, weights, capacity) for ind in population]
        best_fitness = max(fitness_scores)
        log(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
        population = generate_next_generation(population, fitness_scores, mutation_rate, operator_choice)
    fitness_scores = [fitness(ind, values, weights, capacity) for ind in population]
    best_index = np.argmax(fitness_scores)
    return population[best_index], fitness_scores[best_index]

# Parametry algorytmu
num_items = len(weights)
population_size = 6
generations = 30
mutation_rate = 0.1
operator_choice = 0  # 0: Single-point, 1: Two-point

# Uruchomienie algorytmu genetycznego
with open(log_file, "w") as f:
    f.write("Log Start\n")

best_solution, best_solution_fitness = genetic_algorithm(
    values, weights, capacity, num_items, population_size, generations, mutation_rate, operator_choice
)

# Wyświetlenie najlepszego rozwiązania
print("\nNajlepsze rozwiązanie:", best_solution)
print("Wartość fitness najlepszego rozwiązania:", best_solution_fitness)
print("Przedmioty w plecaku:", [i + 1 for i in range(len(best_solution)) if best_solution[i] == 1])
