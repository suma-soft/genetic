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
def tournament_selection(population, fitness_scores, tournament_size=3):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
    return population[best_index]

# Funkcja selekcji proporcjonalnej (metoda ruletki)
def roulette_selection(population, fitness_scores):
    adjusted_fitness = np.array(fitness_scores) - min(fitness_scores) + 1e-6
    probabilities = adjusted_fitness / np.sum(adjusted_fitness)
    chosen_index = np.random.choice(len(population), p=probabilities)
    return population[chosen_index]

# Funkcja selekcji rankingowej
def ranking_selection(population, fitness_scores):
    ranks = np.argsort(np.argsort(fitness_scores)) + 1  # Ranking: najgorszy = 1, najlepszy = n
    probabilities = ranks / np.sum(ranks)  # Prawdopodobieństwa na podstawie rankingu
    chosen_index = np.random.choice(len(population), p=probabilities)
    return population[chosen_index]

# Funkcja krzyżowania jednorodnego
def uniform_crossover(parent1, parent2):
    """
    Krzyżowanie jednorodne.
    Każdy gen potomka jest wybierany losowo z jednego z rodziców.
    """
    mask = np.random.randint(2, size=len(parent1))
    child1 = np.where(mask == 0, parent1, parent2)
    child2 = np.where(mask == 0, parent2, parent1)
    return child1, child2, mask


# Funkcja krzyżowania genów
def crossover(parent1, parent2, operator_choice=1, log_file=None):
    start_time = time.time()

    if operator_choice == 0:  # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        operation = f"Single-point at {crossover_point}"
    elif operator_choice == 1:  # Two-point crossover
        points = sorted(np.random.choice(range(1, len(parent1) - 1), 2, replace=False))
        child1 = np.concatenate((parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]))
        child2 = np.concatenate((parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]))
        operation = f"Two-point at {points}"
    elif operator_choice == 2:  # Uniform crossover
        child1, child2, mask = uniform_crossover(parent1, parent2)
        operation = f"Uniform with mask {mask.tolist()}"
    else:
        raise ValueError("Nieznany operator krzyżowania. Wybierz 0, 1 lub 2.")

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
def generate_next_generation(population, fitness_scores, mutation_rate, selection_method, operator_choice):
    new_population = []
    population_size = len(population)
    for _ in range(population_size // 2):
        if selection_method == "tournament":
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
        elif selection_method == "roulette":
            parent1 = roulette_selection(population, fitness_scores)
            parent2 = roulette_selection(population, fitness_scores)
        elif selection_method == "ranking":
            parent1 = ranking_selection(population, fitness_scores)
            parent2 = ranking_selection(population, fitness_scores)
        else:
            raise ValueError("Nieznana metoda selekcji. Wybierz 'tournament', 'roulette' lub 'ranking'.")

        child1, child2 = crossover(parent1, parent2, operator_choice)
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
def genetic_algorithm(values, weights, capacity, num_items, population_size, generations, mutation_rate, selection_method, operator_choice):
    population = generate_initial_population(num_items, population_size)
    start_time = time.time()  # Rozpoczęcie pomiaru czasu
    log_file = "log.txt"
    diversity_values = []  # Przechowywanie różnorodności dla każdej populacji
    stats = []

    with open(log_file, "w") as log:
        log.write(f"Log algorytmu genetycznego:\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Rozmiar populacji: {population_size}\n")
        log.write(f"Liczba pokoleń: {generations}\n")
        log.write(f"Wskaźnik mutacji: {mutation_rate}\n")
        log.write(f"Metoda selekcji: {selection_method}\n")
        log.write(f"Operator krzyżowania: {'Two-point' if operator_choice == 1 else 'Single-point'}\n")

    for generation in range(generations):
        fitness_scores = [fitness(ind, values, weights, capacity) for ind in population]
        best_fitness = max(fitness_scores)
        worst_fitness = min(fitness_scores)
        average_fitness = np.mean(fitness_scores)
        diversity = calculate_diversity(population)
        stats.append((generation + 1, best_fitness, worst_fitness, average_fitness, diversity))


        with open(log_file, "a") as log:
            log.write(f"\nPokolenie {generation + 1}:\n")
            log.write(f"Najlepsza wartość fitness: {best_fitness}\n")
            log.write(f"Najgorsza wartość fitness: {worst_fitness}\n")
            log.write(f"Średnia wartość fitness: {average_fitness:.2f}\n")
            log.write(f"Różnorodność populacji: {diversity:.4f}\n")

        population = generate_next_generation(population, fitness_scores, mutation_rate, selection_method, operator_choice)

        fitness_scores = [fitness(ind, values, weights, capacity) for ind in population]
        best_index = np.argmax(fitness_scores)
        elapsed_time = time.time() - start_time  # Obliczenie czasu wykonania

    with open("log.txt", "a") as log:
        log.write(f"\nPodsumowanie statystyk dla wszystkich pokoleń:\n")
        log.write(f"{'Pokolenie':<10}{'Najlepsza_Fitness':<20}{'Najgorsza_Fitness':<20}{'Średnia_Fitness':<20}{'Różnorodność':<10}\n")
        for stat in stats:
            log.write(f"{stat[0]:<10}{stat[1]:<20.4f}{stat[2]:<20.4f}{stat[3]:<20.4f}{stat[4]:<10.4f}\n")
        log.write(f"\nCzas wykonania algorytmu: {elapsed_time:.6f} sekund\n")
        log.write(f"\nRozmiar populacji: {population_size}\n")
        log.write(f"Liczba pokoleń: {generations}\n")
        log.write(f"Wskaźnik mutacji: {mutation_rate}\n")
        log.write(f"Metoda selekcji: {selection_method}\n")
        log.write(f"Operator krzyżowania: {['Single-point', 'Two-point', 'Jednorodne'][operator_choice]}\n")


    best_solution = population[best_index]
    best_solution_fitness = fitness_scores[best_index]

    return best_solution, best_solution_fitness

# Parametry algorytmu
num_items = len(weights)
population_size = 10
generations = 100
mutation_rate = 0.05
selection_method = "roulette"  # Możliwość wyboru: "tournament", "roulette", "ranking"
operator_choice = 0  # 0: Single-point, 1: Two-point 2: Jednorodne

# Uruchomienie algorytmu genetycznego
best_solution, best_solution_fitness = genetic_algorithm(
    values, weights, capacity, num_items, population_size, generations, mutation_rate, selection_method, operator_choice
)

# Wyświetlenie najlepszego rozwiązania
print("\nNajlepsze rozwiązanie:", best_solution)
print("Wartość fitness najlepszego rozwiązania:", best_solution_fitness)
print("Przedmioty w plecaku:", [i + 1 for i in range(len(best_solution)) if best_solution[i] == 1])
