import numpy as np

# Krok 1: Przygotowanie danych
# Wejście i wyjście dla operatora XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Wejścia
y = np.array([[0], [1], [1], [0]])  # Oczekiwane wyjścia

# Krok 2: Inicjalizacja sieci
np.random.seed(42)  # Ustawienie ziarna losowości dla powtarzalności wyników
input_neurons = 2  # Liczba neuronów na wejściu
hidden_neurons = 2  # Liczba neuronów w warstwie ukrytej
output_neurons = 1  # Liczba neuronów na wyjściu

# Losowa inicjalizacja wag i biasów
weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))  # Wagi wejście -> ukryta
bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))  # Bias dla warstwy ukrytej

weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))  # Wagi ukryta -> wyjście
bias_output = np.random.uniform(-1, 1, (1, output_neurons))  # Bias dla warstwy wyjściowej

# Hiperparametry
learning_rate = 0.5  # Współczynnik uczenia
epochs = 10000  # Liczba epok

# Krok 3: Funkcja aktywacji i jej pochodna
def sigmoid(x):
    """Funkcja sigmoidalna."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Pochodna funkcji sigmoidalnej, potrzebna do propagacji wstecznej."""
    return x * (1 - x)

# Krok 4: Pętla uczenia
mse_history = []  # Lista do przechowywania wartości MSE
with open("log_xor.txt", "w") as log_file:
    log_file.write("Epoka,MSE\n")  # Nagłówki dla pliku CSV
    for epoch in range(epochs):
        # Propagacja w przód
        # Wejście -> Warstwa ukryta
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden  # Obliczenie sumy ważonej dla warstwy ukrytej
        hidden_output = sigmoid(hidden_input)  # Przejście przez funkcję aktywacji w warstwie ukrytej

        # Warstwa ukryta -> Wyjście
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output  # Obliczenie sumy ważonej dla wyjścia
        final_output = sigmoid(final_input)  # Przejście przez funkcję aktywacji w warstwie wyjściowej

        # Obliczenie błędu
        error = y - final_output  # Różnica między oczekiwanym a uzyskanym wynikiem

        # Propagacja wstecz
        # Gradient dla wyjścia
        output_gradient = error * sigmoid_derivative(final_output)
        weights_hidden_output_update = np.dot(hidden_output.T, output_gradient)  # Aktualizacja wag ukryta -> wyjście
        bias_output_update = np.sum(output_gradient, axis=0, keepdims=True)  # Aktualizacja biasu wyjściowego

        # Gradient dla warstwy ukrytej
        hidden_error = np.dot(output_gradient, weights_hidden_output.T)  # Błąd propagowany do warstwy ukrytej
        hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)  # Gradient w warstwie ukrytej
        weights_input_hidden_update = np.dot(X.T, hidden_gradient)  # Aktualizacja wag wejście -> ukryta
        bias_hidden_update = np.sum(hidden_gradient, axis=0, keepdims=True)  # Aktualizacja biasu ukrytego

        # Aktualizacja wag i biasów
        weights_hidden_output += learning_rate * weights_hidden_output_update
        bias_output += learning_rate * bias_output_update

        weights_input_hidden += learning_rate * weights_input_hidden_update
        bias_hidden += learning_rate * bias_hidden_update

        # Monitorowanie błędu
        mse = np.mean(np.square(error))  # Średni błąd kwadratowy
        mse_history.append(mse)  # Zapisanie MSE do historii
        if (epoch + 1) % 100 == 0:
            log_file.write(f"{epoch + 1},{mse}\n")  # Zapisanie MSE co 100 epok
        if (epoch + 1) % 1000 == 0:
            print(f"Epoka {epoch + 1}, Błąd MSE: {mse}")

# Wagi i biasy po nauce
print("\nWyuczone wagi wejście -> ukryta:", weights_input_hidden)
print("Wyuczony bias ukrytej warstwy:", bias_hidden)
print("Wyuczone wagi ukryta -> wyjście:", weights_hidden_output)
print("Wyuczony bias wyjściowej warstwy:", bias_output)

# Krok 5: Testowanie modelu
print("\nTestowanie modelu:")
predicted_outputs = []  # Lista do przechowywania przewidywanych wyjść
for i, input_data in enumerate(X):
    hidden_input = np.dot(input_data, weights_input_hidden) + bias_hidden  # Wejście -> ukryta
    hidden_output = sigmoid(hidden_input)  # Aktywacja ukrytej warstwy

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output  # Ukryta -> wyjście
    final_output = sigmoid(final_input)  # Aktywacja wyjściowa

    predicted_outputs.append(final_output[0][0])  # Zapisanie wyniku
    print(f"Wejście: {input_data}, Przewidywane wyjście: {final_output[0][0]:.4f}, Oczekiwane wyjście: {y[i][0]}")

# Krok 6: Wizualizacje
# Zmiana błędu MSE w trakcie epok (tekstowa wizualizacja)
print("\nZmiany błędu MSE podczas uczenia:")
for epoch, mse in enumerate(mse_history):
    if (epoch + 1) % 1000 == 0 or epoch == 0 or epoch == len(mse_history) - 1:
        print(f"Epoka {epoch + 1}: Błąd MSE = {mse}")

# Porównanie oczekiwanych i uzyskanych wyników
print("\nPorównanie wyników oczekiwanych i przewidywanych:")
for i, input_data in enumerate(X):
    print(f"Wejście: {input_data}, Oczekiwane: {y[i][0]}, Przewidywane: {predicted_outputs[i]:.4f}")
