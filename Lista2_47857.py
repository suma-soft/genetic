import pandas as pd
import numpy as np

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

#Zrobić kod sprawdzający długość chromosomów


# Funkcja oceny wartości chromosomu (fitness)
def fitness(chromosome, values, weights, capacity):
    total_value = np.sum(chromosome * values)
    total_weight = np.sum(chromosome * weights)
    if total_weight > capacity:
        return 0  # Kara za przekroczenie pojemności plecaka (!!!!! Zbyt surowa) proporcaj między przedziałem wartości a przedziałem wag
    return total_value

# Funkcja generująca losową populację początkową
def generate_initial_population(num_items, population_size):
    return np.random.randint(2, size=(population_size, num_items))

# Funkcja selekcji turniejowej
def tournament_selection(population, fitness_scores, tournament_size=3):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False) #Selekcja bez zwracania jest ok, ale jest szansa, że podczas krzyżowania i tak będą wybrane te same osobniki.
    best_index = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
    return population[best_index]

# Funkcja krzyżowania genów
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1) #!!! Ustalić stały punkt krzyżowania ze względu na krótki wektor binarny
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Funkcja mutacji
def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Inwersja genu
    return chromosome

# Funkcja generująca nową populację
def generate_next_generation(population, fitness_scores, mutation_rate=0.1):
    new_population = []
    population_size = len(population)
    for _ in range(population_size // 2):
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_population.append(child1)
        new_population.append(child2)
    return np.array(new_population)

# Algorytm genetyczny
def genetic_algorithm(values, weights, capacity, num_items, population_size, generations, mutation_rate=0.1):
    population = generate_initial_population(num_items, population_size)
    for generation in range(generations):
        fitness_scores = [fitness(ind, values, weights, capacity) for ind in population]
        best_fitness = max(fitness_scores)
        print(f"Generacja {generation + 1}: Najlepsza wartość fitness = {best_fitness}")
        population = generate_next_generation(population, fitness_scores, mutation_rate)
    fitness_scores = [fitness(ind, values, weights, capacity) for ind in population]
    best_index = np.argmax(fitness_scores)
    return population[best_index], fitness_scores[best_index]

# Parametry algorytmu
num_items = len(weights)
population_size = 6
generations = 30
mutation_rate = 0.1

# Uruchomienie algorytmu genetycznego
best_solution, best_solution_fitness = genetic_algorithm(
    values, weights, capacity, num_items, population_size, generations, mutation_rate
)

# Wyświetlenie najlepszego rozwiązania
print("\nNajlepsze rozwiązanie:", best_solution)
print("Wartość fitness najlepszego rozwiązania:", best_solution_fitness)
print("Przedmioty w plecaku:", [i + 1 for i in range(len(best_solution)) if best_solution[i] == 1])