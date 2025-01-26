import numpy as np
import pandas as pd
from datetime import datetime
import time
import os

# Ustawienie folderu i unikalnej nazwy logu
log_folder = "lista5"
os.makedirs(log_folder, exist_ok=True)  # Tworzenie folderu, jeśli nie istnieje

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_folder, f"log_{current_time}.txt")

# Testowy zapis do logu
with open(log_file, "w") as log:
    log.write("To jest przykładowy log.\n")

print(f"Plik logu: {log_file}")

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

def generate_initial_population_advanced(num_items, population_size, method="uniform"):
    """
    Generuje początkową populację z uwzględnieniem określonego rozkładu.

    Args:
        num_items (int): Liczba przedmiotów (długość chromosomu).
        population_size (int): Liczba osobników w populacji.
        method (str): Metoda losowania: "uniform", "normal", "heuristic".

    Returns:
        np.ndarray: Populacja początkowa.
    """
    if method == "uniform":
        # Jednostajne losowanie (oryginalne podejście)
        return np.random.randint(2, size=(population_size, num_items))
    elif method == "normal":
        # Losowanie według rozkładu normalnego
        population = np.random.normal(0.5, 0.2, size=(population_size, num_items))
        population = np.clip(population, 0, 1)  # Utrzymanie wartości w zakresie [0, 1]
        return (population > 0.5).astype(int)  # Konwersja na wartości binarne
    elif method == "heuristic":
        # Heurystyczne podejście: większe szanse na przedmioty o wysokim stosunku wartości do wagi
        value_to_weight_ratio = values / weights
        probabilities = value_to_weight_ratio / value_to_weight_ratio.sum()
        population = np.random.choice([0, 1], size=(population_size, num_items), p=[1 - probabilities.mean(), probabilities.mean()])
        return population
    else:
        raise ValueError("Nieznana metoda inicjalizacji. Wybierz 'uniform', 'normal' lub 'heuristic'.")


# Funkcja generująca losową populację początkową
def generate_initial_population(num_items, population_size):
    return np.random.randint(2, size=(population_size, num_items))

# Funkcja selekcji turniejowej
def tournament_selection(population, fitness_scores, tournament_size=6): #Tu mozna zmieńić liczbę uczestników turnieju
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
    return population[best_index]

# Funkcja selekcji proporcjonalnej (metoda ruletki)
def roulette_selection(population, fitness_scores):
    adjusted_fitness = np.array(fitness_scores) - min(fitness_scores) + 1e-6
    probabilities = adjusted_fitness / np.sum(adjusted_fitness)
    chosen_index = np.random.choice(len(population), p=probabilities)
    return population[chosen_index]

def adaptive_roulette_selection(population, fitness_scores):
    min_fitness = min(fitness_scores)
    adjusted_fitness = np.array(fitness_scores) - min_fitness + 1e-6  # Ustaw minimalny fitness na 1e-6
    total_fitness = np.sum(adjusted_fitness)
    probabilities = adjusted_fitness / total_fitness  # Oblicz prawdopodobieństwa
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

# Funkcja selekcji rankingowej
def ranking_selection(population, fitness_scores, scale_factor=1.5):
    ranked_indices = np.argsort(fitness_scores)[::-1]  # Sortuj malejąco
    ranks = np.arange(1, len(population) + 1)  # Rangi od 1 do N
    probabilities = (2 - scale_factor) / len(population) + (2 * ranks * (scale_factor - 1)) / (len(population) * (len(population) - 1))
    probabilities /= np.sum(probabilities)  # Normalizacja
    selected_index = np.random.choice(ranked_indices, p=probabilities)
    return population[selected_index]

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

def majority_crossover(parent1, parent2):
    """
    Operator krzyżowania większościowego (Majority Crossover).
    Tworzy potomków na podstawie większości genów obu rodziców.

    Args:
        parent1 (np.ndarray): Pierwszy rodzic.
        parent2 (np.ndarray): Drugi rodzic.

    Returns:
        tuple: Dwóch potomków (child1, child2).
    """
    # Większość genów: Jeśli geny są takie same, zachowujemy je. Jeśli różne, wybieramy losowo.
    child1 = np.where(parent1 == parent2, parent1, np.random.randint(2, size=len(parent1)))
    child2 = np.where(parent1 == parent2, parent2, np.random.randint(2, size=len(parent2)))

    return child1, child2

def mean_crossover(parent1, parent2):
    """
    Operator krzyżowania średniego (Mean Crossover).
    Tworzy potomków na podstawie średniej wartości genów rodziców.

    Args:
        parent1 (np.ndarray): Pierwszy rodzic.
        parent2 (np.ndarray): Drugi rodzic.

    Returns:
        tuple: Dwóch potomków (child1, child2).
    """
    # Obliczanie średniej wartości genów
    child1 = np.round((parent1 + parent2) / 2).astype(int)
    child2 = np.round((parent1 + parent2) / 2).astype(int)

    return child1, child2


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
    elif operator_choice == 2:  # Krzyżowanie jednorodne
        child1, child2, mask = uniform_crossover(parent1, parent2)
        operation = f"Uniform with mask {mask.tolist()}"
    elif operator_choice == 3: # Krzyżowanie średnie
        child1, child2 = mean_crossover(parent1, parent2)
    elif operator_choice == 4: # Krzyżowanie większościowe
        child1, child2 = majority_crossover(parent1, parent2)
    else:
        raise ValueError("Nieznany operator krzyżowania. Wybierz 0-4.")

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
def mutate(chromosome, mutation_rate=0.1, generation=None, max_generations=None, method="standard"):
    """
    Mutacja genów chromosomu.

    Args:
        chromosome (np.ndarray): Chromosom do zmutowania.
        mutation_rate (float): Wskaźnik mutacji (prawdopodobieństwo zmiany genu).
        generation (int): Numer aktualnego pokolenia (dla mutacji dynamicznej).
        max_generations (int): Maksymalna liczba pokoleń (dla mutacji dynamicznej).
        method (str): Metoda mutacji: "standard", "gaussian", "dynamic".

    Returns:
        np.ndarray: Zmutowany chromosom.
    """
    if method == "standard":
        # Standardowa mutacja (inwersja genu)
        for i in range(len(chromosome)):
            if np.random.random() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Zmiana wartości binarnej
    elif method == "gaussian":
        # Mutacja gaussowska (dla wartości rzeczywistych)
        noise = np.random.normal(0, 0.1, size=len(chromosome))  # Dodanie szumu gaussowskiego
        chromosome = np.clip(chromosome + noise, 0, 1)  # Utrzymanie wartości w przedziale [0, 1]
        chromosome = (chromosome > 0.5).astype(int)  # Konwersja na binarne
    elif method == "dynamic":
        # Mutacja dynamiczna (zmieniająca się w zależności od pokolenia)
        if generation is None or max_generations is None:
            raise ValueError("Do dynamicznej mutacji wymagane są generation i max_generations.")
        dynamic_rate = mutation_rate * (1 - (generation / max_generations))  # Zmniejszenie wskaźnika mutacji w czasie
        for i in range(len(chromosome)):
            if np.random.random() < dynamic_rate:
                chromosome[i] = 1 - chromosome[i]
    else:
        raise ValueError("Nieznana metoda mutacji. Wybierz 'standard', 'gaussian' lub 'dynamic'.")
    
    return chromosome

# Funkcja generująca nową populację
def generate_next_generation(population, fitness_scores, mutation_rate, selection_method, operator_choice, mutation_method="standard", generation=None, max_generations=None):
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
            parent1 = ranking_selection(population, fitness_scores, scale_factor=1.7)
            parent2 = ranking_selection(population, fitness_scores, scale_factor=1.7)
        elif selection_method == "adaptive_roulette":
            parent1 = adaptive_roulette_selection(population, fitness_scores)
            parent2 = adaptive_roulette_selection(population, fitness_scores)
        else:
            raise ValueError("Nieznana metoda selekcji. Wybierz 'tournament', 'roulette' lub 'ranking'.")

        child1, child2 = crossover(parent1, parent2, operator_choice)
        child1 = mutate(child1, mutation_rate, generation, max_generations, method=mutation_method)
        child2 = mutate(child2, mutation_rate, generation, max_generations, method=mutation_method)
        new_population.append(child1)
        new_population.append(child2)
    return np.array(new_population)

# Funkcja obliczająca różnorodność genetyczną populacji
def calculate_diversity(population):
    unique_individuals = np.unique(population, axis=0)
    diversity = len(unique_individuals) / len(population)
    return diversity

# Algorytm genetyczny
def genetic_algorithm(values, weights, capacity, num_items, population_size, generations, mutation_rate, selection_method, operator_choice, init_method="uniform", tournament_size=6, scale_factor=1.7, mutation_method="standard"):
    """
    Args:
        values (np.ndarray): Wartości przedmiotów.
        weights (np.ndarray): Wagi przedmiotów.
        capacity (int): Pojemność plecaka.
        num_items (int): Liczba przedmiotów.
        population_size (int): Rozmiar populacji.
        generations (int): Liczba pokoleń.
        mutation_rate (float): Wskaźnik mutacji.
        selection_method (str): Metoda selekcji: "tournament", "roulette", "ranking", "adaptive_roulette".
        operator_choice (int): Wybór operatora krzyżowania.
        init_method (str): Metoda inicjalizacji populacji: "uniform", "normal", "heuristic".
        tournament_size (int): Rozmiar turnieju (dla selekcji turniejowej).
        scale_factor (float): Współczynnik skali (dla selekcji rankingowej).
        mutation_method (str): Metoda mutacji: "standard", "gaussian", "dynamic".
    """
    population = generate_initial_population_advanced(num_items, population_size, method=init_method)
    start_time = time.time()  # Rozpoczęcie pomiaru czasu
    stats = []

    with open(log_file, "w") as log:
        log.write(f"Log algorytmu genetycznego:\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Rozmiar populacji: {population_size}\n")
        log.write(f"Liczba pokoleń: {generations}\n")
        log.write(f"Wskaźnik mutacji: {mutation_rate}\n")
        log.write(f"Metoda selekcji: {selection_method}\n")
        log.write(f"Operator krzyżowania: {'Two-point' if operator_choice == 1 else 'Single-point'}\n")
        log.write(f"Metoda mutacji: {mutation_method}\n")
        log.write(f"Metoda inicjalizacji populacji: {init_method}\n")

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

        population = generate_next_generation(
            population, fitness_scores, mutation_rate, selection_method, operator_choice,
            mutation_method=mutation_method, generation=generation, max_generations=generations
        )

    elapsed_time = time.time() - start_time  # Obliczenie czasu wykonania
    best_index = np.argmax(fitness_scores)
    best_solution = population[best_index]
    best_solution_fitness = fitness_scores[best_index]

    with open(log_file, "a") as log:
        log.write(f"\nPodsumowanie statystyk dla wszystkich pokoleń:\n")
        log.write(f"{'Pokolenie':<10}{'Najlepsza_Fitness':<20}{'Najgorsza_Fitness':<20}{'Średnia_Fitness':<20}{'Różnorodność':<10}\n")
        for stat in stats:
            log.write(f"{stat[0]:<10}{stat[1]:<20.4f}{stat[2]:<20.4f}{stat[3]:<20.4f}{stat[4]:<10.4f}\n")
        log.write(f"\nCzas wykonania algorytmu: {elapsed_time:.6f} sekund\n")
        log.write(f"\nRozmiar populacji: {population_size}\n")
        log.write(f"Liczba pokoleń: {generations}\n")
        log.write(f"Wskaźnik mutacji: {mutation_rate}\n")
        log.write(f"Metoda selekcji: {selection_method}\n")
        if selection_method == "tournament":
            log.write(f"Rozmiar turnieju: {tournament_size}\n")
        elif selection_method == "ranking":
            log.write(f"Współczynnik skali rankingowej: {scale_factor}\n")
        log.write(f"Operator krzyżowania: {['Single-point', 'Two-point', 'Jednorodne', 'Średnie', 'większościowe'][operator_choice]}\n")
        log.write(f"Metoda mutacji: {mutation_method}\n")
        log.write(f"Metoda inicjalizacji populacji: {init_method}\n")
        log.write(f"\n\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return best_solution, best_solution_fitness


def new_func():
    log_file = "log_file"
    return log_file

# Parametry algorytmu
num_items = len(weights)
population_size = 10
generations = 10
mutation_rate = 0.05
selection_method = "roulette"  # Możliwość wyboru: "tournament", "roulette", "ranking", "adaptive_roulette"
operator_choice = 0  # 0: Single-point, 1: Two-point, 2: Jednorodne, 3: Krzyżowanie średnie
mutation_method = "gaussian"  # Możliwość wyboru: "standard", "gaussian", "dynamic"

# Uruchomienie algorytmu genetycznego
best_solution, best_solution_fitness = genetic_algorithm(
    values, weights, capacity, num_items, population_size, generations,
    mutation_rate, selection_method, operator_choice, mutation_method="dynamic"
)

# Wyświetlenie najlepszego rozwiązania
print("\nNajlepsze rozwiązanie:", best_solution)
print("Wartość fitness najlepszego rozwiązania:", best_solution_fitness)
print("Przedmioty w plecaku:", [i + 1 for i in range(len(best_solution)) if best_solution[i] == 1])
