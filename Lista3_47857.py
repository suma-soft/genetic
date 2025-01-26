import numpy as np
import pandas as pd
from datetime import datetime
import time

# Wczytanie danych z pliku zbior_danych_ag.csv
data = pd.read_csv("zbior_danych_ag.csv", sep=';')

# Funkcja przetwarzająca tekst na tablicę numpy
def parse_array(array_string):
    array_string = array_string.strip("[]")  # Usuń nawiasy []
    array_string = array_string.replace("  ", " ")  # Zredukuj podwójne spacje
    elements = [int(x) for x in array_string.split()]  # Podziel ciąg po spacjach i konwertuj na liczby
    return np.array(elements) #Zamień listy elementów na obiekty tablicy numpy

# Zamiana kolumn na tablice numpy
data['Ciezar'] = data['Ciezar'].apply(parse_array) #Zamień str danych na tablice
data['Ceny'] = data['Ceny'].apply(parse_array)

# Pobranie wag, wartości i pojemności dla pierwszego zestawu danych
weights = data['Ciezar'][0] # np.array([46, 40, 42, 38, 10])
values = data['Ceny'][0]  # np.array([12, 19, 19, 15, 8])
capacity = data['Pojemnosc'][0] # 40

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

# Funkcja generująca losową populację początkową
def generate_initial_population(num_items, population_size):
    return np.random.randint(2, size=(population_size, num_items))

# Funkcja selekcji turniejowej
def tournament_selection(population, fitness_scores, tournament_size=3, log_file=None, used_tournaments=set()):

    while True:
        # Losowy wybór indeksów osobników do turnieju
        selected_indices = tuple(map(int, np.random.choice(len(population), tournament_size, replace=False)))
        if selected_indices not in used_tournaments:
            used_tournaments.add(selected_indices)
            break

    selected_chromosomes = [population[i].tolist() for i in selected_indices]
    selected_fitness = [round(float(fitness_scores[i]), 2) for i in selected_indices]
    
    # Wybór najlepszego osobnika
    best_index = selected_indices[np.argmax(selected_fitness)]
    best_individual = population[best_index]

    # Logowanie wyników turnieju
    if log_file:
        with open(log_file, "a") as log:
            log.write(f"\nTurniej: {list(selected_indices)}\n")
            log.write(f"Chromosomy w turnieju: {selected_chromosomes}\n")
            log.write(f"Fitness wybranych: {selected_fitness}\n")
            log.write(f"Wybrany osobnik: {best_individual.tolist()}, Fitness: {round(float(fitness_scores[best_index]), 2)}\n")

    return best_individual

# Funkcja selekcji proporcjonalnej (metoda ruletki)
def roulette_selection(population, fitness_scores, log_file=None):

    adjusted_fitness = np.array(fitness_scores) - min(fitness_scores) + 1e-6
    probabilities = adjusted_fitness / np.sum(adjusted_fitness)
    chosen_index = int(np.random.choice(len(population), p=probabilities))
    chosen_individual = population[chosen_index]

    if log_file:
        with open(log_file, "a") as log:
            log.write(f"\nRuletka:\n")
            log.write(f"Fitness: {[round(f, 2) for f in fitness_scores]}\n")
            log.write(f"Prawdopodobieństwa: {[round(p, 4) for p in probabilities]}\n")
            log.write(f"Wybrany osobnik: {chosen_individual.tolist()}, Fitness: {round(float(fitness_scores[chosen_index]), 2)}\n")

    return chosen_individual

# Funkcja krzyżowania genów
def crossover(parent1, parent2, operator_choice=1, log_file=None):
    start_time = time.time()

    if operator_choice == 0:
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        operation = f"Single-point at {crossover_point}"
    elif operator_choice == 1:
        points = sorted(np.random.choice(range(1, len(parent1) - 1), 2, replace=False))
        child1 = np.concatenate((parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]))
        child2 = np.concatenate((parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]))
        operation = f"Two-point at {points}"
    else:
        raise ValueError("Nieznany operator krzyżowania. Wybierz 0 lub 1.")

    if log_file:
        elapsed_time = time.time() - start_time
        with open(log_file, "a") as log:
            log.write(f"\nKrzyżowanie ({operation}, czas: {elapsed_time:.6f} s):\n")
            log.write(f"Rodzic 1: {parent1.tolist()}\n")
            log.write(f"Rodzic 2: {parent2.tolist()}\n")
            log.write(f"Dziecko 1: {child1.tolist()}\n")
            log.write(f"Dziecko 2: {child2.tolist()}\n")

    return child1, child2

# Funkcja mutacji
def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Inwersja genu
    return chromosome

# Funkcja generująca nową populację
def generate_next_generation(population, fitness_scores, mutation_rate=0.1, selection_method="tournament", operator_choice=1, log_file="log.txt"):
    new_population = []
    population_size = len(population)
    used_tournaments = set()
    for _ in range(population_size // 2):
        if selection_method == "tournament":
            parent1 = tournament_selection(population, fitness_scores, log_file=log_file, used_tournaments=used_tournaments)
            parent2 = tournament_selection(population, fitness_scores, log_file=log_file, used_tournaments=used_tournaments)
        elif selection_method == "roulette":
            parent1 = roulette_selection(population, fitness_scores, log_file=log_file)
            parent2 = roulette_selection(population, fitness_scores, log_file=log_file)
        else:
            raise ValueError("Nieznana metoda selekcji. Wybierz 'tournament' lub 'roulette'.")

        child1, child2 = crossover(parent1, parent2, operator_choice, log_file=log_file)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_population.append(child1)
        new_population.append(child2)
    return np.array(new_population)

# Funkcja obliczająca różnorodność genetyczną populacji
def calculate_diversity(population):
    unique_individuals = np.unique(population, axis=0)
    diversity = len(unique_individuals) / len(population)
    return diversity

# Algorytm genetyczny
def genetic_algorithm(values, weights, capacity, num_items, population_size, generations, mutation_rate=0.1, selection_method="tournament", operator_choice=1):
    start_time = time.time()
    diversity_values = []  # Przechowywanie różnorodności dla każdej populacji

    with open("log.txt", "w") as log:
        log.write(f"Log algorytmu genetycznego:\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Rozmiar populacji: {population_size}\n")
        log.write(f"Liczba pokoleń: {generations}\n")
        log.write(f"Wskaźnik mutacji: {mutation_rate}\n")
        log.write(f"Metoda selekcji: {selection_method}\n")
        log.write(f"Operator krzyżowania: {'Two-point' if operator_choice == 1 else 'Single-point'}\n")

    population = generate_initial_population(num_items, population_size)
    for generation in range(generations):
        fitness_scores = [fitness(ind, values, weights, capacity) for ind in population]
        best_fitness = max(fitness_scores)
        diversity = calculate_diversity(population)  # Oblicz różnorodność populacji
        diversity_values.append(diversity)

        with open("log.txt", "a") as log:
            log.write(f"\nPokolenie {generation + 1}:\n")
            log.write(f"Najlepsza wartość fitness: {best_fitness}\n")
            log.write(f"Średnia różnorodność populacji: {diversity:.4f}\n")

        print(f"Generacja {generation + 1}: Najlepsza wartość fitness = {best_fitness}, Różnorodność = {diversity:.4f}")
        population = generate_next_generation(population, fitness_scores, mutation_rate, selection_method, operator_choice, log_file="log.txt")

    fitness_scores = [fitness(ind, values, weights, capacity) for ind in population]
    best_index = np.argmax(fitness_scores)

    elapsed_time = time.time() - start_time
    best_solution = population[best_index]
    best_solution_fitness = fitness_scores[best_index]
    average_diversity = np.mean(diversity_values)  # Oblicz średnią różnorodność populacji

    with open("log.txt", "a") as log:
        log.write(f"\nCzas wykonania algorytmu: {elapsed_time:.6f} sekund\n")
        log.write(f"Średnia różnorodność wszystkich populacji: {average_diversity:.4f}\n")
        log.write(f"\nNajlepsze rozwiązanie: {best_solution.tolist()}\n")
        log.write(f"Wartość fitness najlepszego rozwiązania: {best_solution_fitness}\n")
        log.write(f"Przedmioty w plecaku: {[i + 1 for i in range(len(best_solution)) if best_solution[i] == 1]}\n")

    return best_solution, best_solution_fitness

# Parametry algorytmu
num_items = len(weights)
population_size = 6
generations = 10
mutation_rate = 0.05
selection_method = "tournament"  # Możliwość zmiawyboruny  "tournament" lub "roulette"
operator_choice = 1  # 0: Single-point, 1: Two-point

# Uruchomienie algorytmu genetycznego
best_solution, best_solution_fitness = genetic_algorithm(
    values, weights, capacity, num_items, population_size, generations, mutation_rate, selection_method, operator_choice
)

# Wyświetlenie najlepszego rozwiązania
print("\nNajlepsze rozwiązanie:", best_solution)
print("Wartość fitness najlepszego rozwiązania:", best_solution_fitness)
print("Przedmioty w plecaku:", [i + 1 for i in range(len(best_solution)) if best_solution[i] == 1])
