import numpy as np

# Krok 1: Przygotowanie danych
# Wejście i wyjście dla operatora XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Wejścia
y = np.array([[0], [1], [1], [0]])  # Oczekiwane wyjścia

# Krok 2: Inicjalizacja sieci
np.random.seed(42)  # Ustawienie ziarna losowości dla powtarzalności wyników
input_neurons = 2  # Liczba neuronów na wejściu
output_neurons = 1  # Liczba neuronów na wyjściu

# Losowa inicjalizacja wag i biasów
weights = np.random.uniform(-1, 1, (input_neurons, output_neurons))  # Macierz wag
bias = np.random.uniform(-1, 1, (1, output_neurons))  # Wektor biasów

# Hiperparametry
learning_rate = 0.1  # Współczynnik uczenia
epochs = 10000  # Liczba epok

# Krok 3: Funkcja aktywacji i jej pochodna
def sigmoid(x):
    """Funkcja sigmoidalna."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Pochodna funkcji sigmoidalnej, potrzebna do propagacji wstecznej."""
    return x * (1 - x)

# Krok 4: Pętla uczenia
for epoch in range(epochs):
    # Propagacja w przód
    weighted_sum = np.dot(X, weights) + bias  # Obliczenie sumy ważonej
    output = sigmoid(weighted_sum)  # Przejście przez funkcję aktywacji

    # Obliczenie błędu
    error = y - output  # Różnica między oczekiwanym a uzyskanym wynikiem

    # Propagacja wstecz
    gradient = error * sigmoid_derivative(output)  # Wyliczenie gradientu
    weights_update = np.dot(X.T, gradient)  # Aktualizacja wag na podstawie gradientu
    
    # Aktualizacja wag i biasów
    weights += learning_rate * weights_update
    bias += learning_rate * np.sum(gradient, axis=0, keepdims=True)

    # Monitorowanie błędu co 1000 epok
    if (epoch + 1) % 1000 == 0:
        mse = np.mean(np.square(error))  # Średni błąd kwadratowy
        print(f"Epoka {epoch + 1}, Błąd MSE: {mse}")

# Wagi i biasy po nauce
print("\nWyuczone wagi:", weights)
print("Wyuczony bias:", bias)

# Testowanie modelu
print("\nTestowanie modelu:")
for i, input_data in enumerate(X):
    result = sigmoid(np.dot(input_data, weights) + bias)  # Obliczenie wyjścia modelu
    print(f"Wejście: {input_data}, Przewidywane wyjście: {result[0][0]:.4f}, Oczekiwane wyjście: {y[i][0]}")
