import quantum_kan 
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import precision_score, recall_score, f1_score
from pyqubo import Binary, Array
import os
from dimod import BinaryQuadraticModel
# Solve the QUBO using a quantum annealer
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
import time 
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from qkan import KAN
import torch.nn as nn

# Set your D-Wave API token
import os
# env_vars = !cat dwave.env
# for var in env_vars:
#     key, value = var.split('=')
#     os.environ[str(key)] = str(value)
# sampler = EmbeddingComposite(DWaveSampler(solver='Advantage2_prototype2.3'))
simulated_sampler = SimulatedAnnealingSampler()

# Set the number of threads for intra-op parallelism (used for CPU operations)
torch.set_num_threads(1)

# Set the number of threads for inter-op parallelism (used for parallelism between operations)
torch.set_num_interop_threads(1)

# Verify the settings
print(f"Number of intra-op threads: {torch.get_num_threads()}")
print(f"Number of inter-op threads: {torch.get_num_interop_threads()}")

def convert_sympy_to_pyqubo(objective_func, conversion_dict):
    # Parse the simplified symbolic equation and update the QUBO matrix
    
    expression = mse_with_penalty_str
    pyqubo_expr = 0
    # Replace '-' with '+-' to simplify splitting
    expression = expression.replace(' - ', ' + -')
    # Split the expression into terms
    terms = expression.split(' + ')
    pyqubo_obj = 0

    for term in terms:
        # Split term into coefficient and variable parts
        parts = term.split('*')
        current_pyqubo_term = int(0)
        first_var = True
        for part in parts:
            if ('P' or 'AUX') in part:
                current_pyqubo_var = conversion_dict.get(part)
            else:
                current_pyqubo_var = float(part)
                
            if first_var:
                current_pyqubo_term = current_pyqubo_var
                first_var = False
            else:
                current_pyqubo_term = current_pyqubo_term * current_pyqubo_var
        # current_pyqubo_term = current_pyqubo_term * coeff
        pyqubo_obj += current_pyqubo_term
        del current_pyqubo_term
        
    return pyqubo_obj

# Define the Bernstein polynomial basis functions
def bernstein_basis_functions(x, degree):
    """
    Calculates the Bernstein polynomial basis functions.

    Args:
        x (array-like): Points at which to evaluate the basis functions.
        degree (int): Degree of the Bernstein polynomial.

    Returns:
        numpy.ndarray: A 2D array of shape (n_basis, len(x)), where each row 
                       represents a basis function evaluated at the points in x.
    """
    n = degree
    basis = np.zeros((n+1, len(x)))

    for i in range(n+1):
        binomial_coeff = math.comb(n, i)
        basis[i, :] = binomial_coeff * (x**i) * ((1 - x)**(n - i))

    return basis

def bernstein_poly(i, n, t):
    """Compute the Bernstein polynomial B_i^n at t."""
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(coefficients, t):
    """Compute a Bezier curve given coefficients and parameter t."""
    n = len(coefficients) - 1
    return sum(coefficients[i] * bernstein_poly(i, n, t) for i in range(n + 1))

# Normalize x_data and y_data
def normalize(data):
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min)


# Define a custom base function that always returns zero
class ZeroBaseFunction(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

def train_classical_optimizer(dataset, degrees, optimizer, steps, learning_rates, x_data_test, y_data_test, z_data_train, z_data_test):
    best_lr = None
    best_acc = 0


    def train_acc():
        return torch.mean((torch.round(model(dataset['train_input'])[:, 0]) == dataset['train_label'][:, 0]).float())

    def test_acc():
        return torch.mean((torch.round(model(dataset['test_input'])[:, 0]) == dataset['test_label'][:, 0]).float())


    for lr in learning_rates:
        model = KAN(width=[2, 1], degrees=degrees, bias_trainable=False, base_fun=ZeroBaseFunction()).float()
        start = time.perf_counter()
        results = model.train(dataset, opt=optimizer, steps=steps, log=10, lr=lr, metrics=(train_acc, test_acc))
        end = time.perf_counter()
        print(f"Training time in seconds of lr={lr}: {end - start}")  # will print the elapsed time in seconds

        if results['test_acc'][-1] > best_acc:
            best_acc = results['test_acc'][-1]
            best_lr = lr
        print(f"Learning Rate: {lr}, Train Accuracy: {results['train_acc'][-1]}, Test Accuracy: {results['test_acc'][-1]}")

    print(f"Best Learning Rate: {best_lr} with Test Accuracy: {best_acc}")

    # Plot the results using the best learning rate
    model = KAN(width=[2, 1], degrees=degrees, bias_trainable=False, base_fun=ZeroBaseFunction()).float()
    start = time.perf_counter()
    results = model.train(dataset, opt=optimizer, steps=int(steps), log=10, lr=best_lr, metrics=(train_acc, test_acc))
    end = time.perf_counter()
    time_best_lr = end - start
    print(f"Training time in seconds of best lr which is lr={lr}: {time_best_lr}")  # will print the elapsed time in seconds
    print(f"Best Learning Rate: {best_lr}, Train Accuracy: {results['train_acc'][-1]}, Test Accuracy: {results['test_acc'][-1]}")

    # Print final Bezier coefficients for each layer
    for layer_idx, layer in enumerate(model.act_fun):
        print(f"Layer {layer_idx} Bezier coefficients:")
        print(layer.coef.data.cpu().numpy())

    # Generate predictions
    with torch.no_grad():
        predictions = model(dataset['test_input']).numpy().flatten()
        thresholded_predictions = np.where(predictions > 0.5, 1, 0)

    plt.scatter(x_data_test[thresholded_predictions == 0], y_data_test[thresholded_predictions == 0], label='Class 0', c='blue', alpha=0.6)
    plt.scatter(x_data_test[thresholded_predictions == 1], y_data_test[thresholded_predictions == 1], label='Class 1', c='orange', alpha=0.6)
    # Add legend to the plot
    plt.legend()
    plt.xlabel('Normalized X Data')
    plt.ylabel('Normalized Y Data')
    plt.title('Scatter Plot with Predicted Values from KAN Model using the Adam Optimizer')
    plt.show()

    model.plot()

    # Generate predictions
    with torch.no_grad():
        predictions = model(dataset['train_input']).numpy().flatten()
        thresholded_predictions = np.where(predictions > 0.5, 1, 0)

    # Calculate precision, recall, and F1 scores
    train_accuracy = torch.mean((torch.tensor(thresholded_predictions) == z_data_train).float())
    train_precision = precision_score(dataset['train_label'].numpy(), thresholded_predictions)
    train_recall = recall_score(dataset['train_label'].numpy(), thresholded_predictions)
    train_f1 = f1_score(dataset['train_label'].numpy(), thresholded_predictions)

    # Generate predictions
    with torch.no_grad():
        predictions = model(dataset['test_input']).numpy().flatten()
        thresholded_predictions = np.where(predictions > 0.5, 1, 0)

    # Calculate precision, recall, and F1 scores
    test_accuracy = torch.mean((torch.tensor(thresholded_predictions) == z_data_test).float())
    test_precision = precision_score(dataset['test_label'].numpy(), thresholded_predictions)
    test_recall = recall_score(dataset['test_label'].numpy(), thresholded_predictions)
    test_f1 = f1_score(dataset['test_label'].numpy(), thresholded_predictions)
    print(f"time_best_lr: {time_best_lr}, train_f1: {train_f1}, test_f1: {test_f1}")
    
    return time_best_lr, train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1

def run_annealing(type, sampler, problem, chain_strength, time_part1, num_basis_funcs_1, num_basis_funcs_2, m1, m2, degree1, degree2, x_data_train, y_data_train, z_data_train, x_data_test, y_data_test, z_data_test, dataset, num_reads=1000, annealing_time=20):
    #Check if simulated annealer, hybrid simulated annealer, or simulated annealer
    if type == "QA":
        response = sampler.sample(problem, num_reads=num_reads, chain_strength=chain_strength, annealing_time=annealing_time)
        # Extract and print timing information
        timing_info = response.info['timing']

        print("Timing Information:")
        for key, value in timing_info.items():
            print(f"{key}: {value} microseconds")

        # Specific time spent on the simulated processor
        qpu_access_time = timing_info.get('qpu_access_time', 'N/A')
        print(f"\nQPU Access Time: {qpu_access_time} microseconds")

        time_part2 = qpu_access_time * (10**(-6))
        print(f"Time spent on simulated annealer part 2: {time_part2}")  # will print the elapsed time in seconds

    elif type == "HQA":
        response = sampler.sample(problem)
        # Extract and print timing information
        timing_info = response.info

        # Print timing information
        print("Timing Information:")
        for key, value in timing_info.items():
            print(f"{key}: {value} microseconds")

        # Specific time spent on the quantum processor
        qpu_access_time = timing_info.get('qpu_access_time', 'N/A')
        print(f"\nQPU Access Time: {qpu_access_time} microseconds")

        dwave_run_time = timing_info.get('run_time', 'N/A')
        print(f"\nTotal D-Wave Run Time Including Setup: {dwave_run_time} microseconds")

        time_part2 = dwave_run_time * (10**(-6))
        print(f"Time spent on hybrid part 2: {time_part2}")  # will print the elapsed time in seconds

    elif type == "SA":
        # Simulated annealing doesnt converge to a good solution as fast so generally takes more reads
        start_time_part2 = time.perf_counter()
        response = sampler.sample_qubo(problem, num_reads=num_reads)
        end_time_part2 = time.perf_counter()
        time_part2 = end_time_part2 - start_time_part2
        print(f"Time spent on simulated part 2: {time_part2}")  # will print the elapsed time in seconds
    
    # Start part 3
    start_time_part3 = time.perf_counter()
    solution = response.first.sample
    # Extract optimized coefficients
    optimized_coefficients_1 = []
    for i in range(num_basis_funcs_1):
        coeff_value_1 = sum(2**l * solution[f'coeff_plus_1[{i}][{l + m1}]'] - 2**l * solution[f'coeff_minus_1[{i}][{l + m1}]'] for l in range(-m1, m1 + 1))
        optimized_coefficients_1.append(coeff_value_1)
        
    optimized_coefficients_2 = []
    for i in range(num_basis_funcs_2):
        coeff_value_2 = sum(2**l * solution[f'coeff_plus_2[{i}][{l + m2}]'] - 2**l * solution[f'coeff_minus_2[{i}][{l + m2}]'] for l in range(-m2, m2 + 1))
        optimized_coefficients_2.append(coeff_value_2)
        
    end_time_part3 = time.perf_counter()

    time_part3 = end_time_part3 - start_time_part3
    print(f"Time spent on part 3: {time_part3}")  # will print the elapsed time in seconds

    total_time_optimization = time_part1 + time_part2 + time_part3
    print(f"Time spent on optimization: {total_time_optimization}")  # will print the elapsed time in seconds

    # Calculate Bernstein polynomial values on a fine grid
    num_fine_grids = 1000
    x_fine = np.linspace(0, 1, num_fine_grids)
    basis_functions_fine1 = bernstein_basis_functions(x_fine / 1.0, degree1)  # Normalize x to [0, 1]
    spline_values1 = np.zeros_like(x_fine)

    for i in range(num_basis_funcs_1):
        spline_values1 += optimized_coefficients_1[i] * basis_functions_fine1[i]
        
    # Plot the original data points and the fitted Bernstein polynomial spline
    plt.plot(x_fine, spline_values1, label='Fitted Bernstein polynomial spline1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Quantum Optimized Bernstein Polynomial Spline Fit of Bezier 1')
    plt.show()

    # Calculate Bernstein polynomial values on a fine grid
    basis_functions_fine2 = bernstein_basis_functions(x_fine / 1.0, degree2)  # Normalize x to [0, 1]
    spline_values2 = np.zeros_like(x_fine)

    for i in range(num_basis_funcs_2):
        spline_values2 += optimized_coefficients_2[i] * basis_functions_fine2[i]
        
    # Plot the original data points and the fitted Bernstein polynomial spline
    plt.plot(x_fine, spline_values2, label='Fitted Bernstein polynomial spline2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Quantum Optimized Bernstein Polynomial Spline Fit of Bezier 2')
    plt.show()

    # Calculate the bezier curve values
    bezier_result = bezier_curve(optimized_coefficients_1, x_data_test) + bezier_curve(optimized_coefficients_2, y_data_test)
    thresholded_bezier_result = np.where(bezier_result > 0.5, 1, 0)

    # Plot the data points with different colors for each label
    plt.scatter(x_data_test[thresholded_bezier_result == 0], y_data_test[thresholded_bezier_result == 0], label='Class 0', c='blue', alpha=0.6)
    plt.scatter(x_data_test[thresholded_bezier_result == 1], y_data_test[thresholded_bezier_result == 1], label='Class 1', c='orange', alpha=0.6)

    # Add legend to the plot
    plt.legend()

    # Show the plot
    plt.xlabel('Normalized X Data')
    plt.ylabel('Normalized Y Data')
    plt.title('Scatter Plot with Bezier Curve Result')
    plt.show()

    x_data_train_np = np.array(x_data_train)
    y_data_train_np = np.array(y_data_train)
    x_data_test_np = np.array(x_data_test)
    y_data_test_np = np.array(y_data_test)

    predicted_labels_train = []
    for i in range(len(x_data_train_np)):
        predicted_value_train = bezier_curve(optimized_coefficients_1, x_data_train_np[i]) + bezier_curve(optimized_coefficients_2, y_data_train_np[i])
        predicted_labels_train.append(int(np.where(predicted_value_train > 0.5, 1, 0)))
        
    predicted_labels_test = []
    for i in range(len(x_data_test_np)):
        predicted_value_test = bezier_curve(optimized_coefficients_1, x_data_test_np[i]) + bezier_curve(optimized_coefficients_2, y_data_test_np[i])
        predicted_labels_test.append(int(np.where(predicted_value_test > 0.5, 1, 0)))


    # Calculate precision, recall, and F1 scores
    train_accuracy = torch.mean((torch.tensor(predicted_labels_train) == z_data_train).float())
    train_precision = precision_score(dataset['train_label'].numpy(), predicted_labels_train)
    train_recall = recall_score(dataset['train_label'].numpy(), predicted_labels_train)
    train_f1 = f1_score(dataset['train_label'].numpy(), predicted_labels_train)

    # Calculate precision, recall, and F1 scores
    test_accuracy = torch.mean((torch.tensor(predicted_labels_test) == z_data_test).float())
    test_precision = precision_score(dataset['test_label'].numpy(), predicted_labels_test)
    test_recall = recall_score(dataset['test_label'].numpy(), predicted_labels_test)
    test_f1 = f1_score(dataset['test_label'].numpy(), predicted_labels_test)

    print(f"total_time_optimization: {total_time_optimization}, train_f1: {train_f1}, test_f1: {test_f1}")
    
    return time_part2, time_part3, total_time_optimization, train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1

# Defining variables for the networks
degree1 = 3
degree2 = 3
degrees=[degree1,degree2]
m1 = 3
m2 = 1
penalty_multiplier = 15
bias_coefficient = 0.0
test_multiplier = 0.9
num_basis_funcs_1 = degree1 + 1
num_basis_funcs_2 = degree2 + 1

# Generate example data
dataset = {}
dataset_np = {}
# n_training_samples = 100000 #5
# n_test_samples = 10000 #5
n_training_samples = 5
n_test_samples = 5
train_input, train_label = make_moons(n_samples=n_training_samples, shuffle=False, noise=0.0, random_state=None)
test_input, test_label = make_moons(n_samples=n_test_samples, shuffle=False, noise=0.0, random_state=None)



dataset_np['train_input'] = torch.from_numpy(train_input)
dataset_np['test_input'] = torch.from_numpy(test_input)
dataset_np['train_label'] = torch.from_numpy(train_label[:, None])
dataset_np['test_label'] = torch.from_numpy(test_label[:, None])

print(f"dataset_np['train_input']: {dataset_np['train_input']}")
print(f"sum X, Y: {np.sum(train_input, axis=0)}")
print(f"sum Z: {np.sum(train_label[:, None])}")

print(f"dataset_np['train_label']: {dataset_np['train_label']}")
print(f"dataset_np['test_input']: {dataset_np['test_input']}")
print(f"dataset_np['test_label']: {dataset_np['test_label']}")

# Convert the data to torch.float
train_input = torch.from_numpy(train_input).float()
test_input = torch.from_numpy(test_input).float()

# Normalize the data to be between 0 and 1
train_min = train_input.min(dim=0, keepdim=True)[0]
train_max = train_input.max(dim=0, keepdim=True)[0]
test_min = test_input.min(dim=0, keepdim=True)[0]
test_max = test_input.max(dim=0, keepdim=True)[0]

train_input = (train_input - train_min) / (train_max - train_min)
test_input = (test_input - test_min) / (test_max - test_min)

dataset['train_input'] = train_input
dataset['test_input'] = test_input
dataset['train_label'] = torch.from_numpy(train_label[:, None]).float()
dataset['test_label'] = torch.from_numpy(test_label[:, None]).float()

X_train = dataset_np['train_input']
y_train = dataset_np['train_label']
z_data_train = y_train[:, 0]
x_data_train = X_train[:, 0]
y_data_train = X_train[:, 1]

X_test = dataset_np['test_input']
y_test = dataset_np['test_label']
z_data_test = y_test[:, 0]
x_data_test = X_test[:, 0]
y_data_test = X_test[:, 1]

# Normalize x_data
x_min_train, x_max_train = x_data_train.min(), x_data_train.max()
x_data_train = (x_data_train - x_min_train) / (x_max_train - x_min_train)

# Normalize y_data
y_min_train, y_max_train = y_data_train.min(), y_data_train.max()
y_data_train = (y_data_train - y_min_train) / (y_max_train - y_min_train)

# Normalize x_data test
x_min_test, x_max_test = x_data_test.min(), x_data_test.max()
x_data_test = (x_data_test - x_min_test) / (x_max_test - x_min_test)

# Normalize y_data test
y_min_test, y_max_test = y_data_test.min(), y_data_test.max()
y_data_test = (y_data_test - y_min_test) / (y_max_test - y_min_test)

plt.scatter(x_data_train, y_data_train, c=z_data_train)

start_time_quantum_part1 = time.perf_counter()
# Call the C++ function and unpack the returned values
result = quantum_kan.compute_mse_with_penalty_categorical(
    d1=degree1, d2=degree2,
    m1=m1, m2=m2,
    penalty_multiplier=penalty_multiplier,
    bias_coefficient=bias_coefficient,
    x_data_train=x_data_train.tolist(),
    y_data_train=y_data_train.tolist(),
    z_data_train=z_data_train.tolist(),
    x_data_test=x_data_test.tolist(),
    y_data_test=y_data_test.tolist(),
    z_data_test=z_data_test.tolist(),
    test_multiplier=test_multiplier
)
print(f"x_data_train.tolist(): {x_data_train.tolist()}")
print(f"sum x_data_train.tolist(): {sum(x_data_train.tolist())}")
print(f"y_data_train.tolist(): {y_data_train.tolist()}")
print(f"sum y_data_train.tolist(): {sum(y_data_train.tolist())}")
print(f"z_data_train.tolist(): {z_data_train.tolist()}")
print(f"sum z_data_train.tolist(): {sum(z_data_train.tolist())}")
mse_with_penalty_str, aux_dict_str, coeffs_plus1_str, coeffs_minus1_str, coeffs_plus2_str, coeffs_minus2_str = result

flattened_coeffs_plus1_str = [item for sublist in coeffs_plus1_str for item in sublist]
flattened_coeffs_minus1_str = [item for sublist in coeffs_minus1_str for item in sublist]
flattened_coeffs_plus2_str = [item for sublist in coeffs_plus2_str for item in sublist]
flattened_coeffs_minus2_str = [item for sublist in coeffs_minus2_str for item in sublist]

# Define P variables (flatten the matrix of Binary variables to a list)
Solving_for_vars = flattened_coeffs_plus1_str[:] + flattened_coeffs_minus1_str[:] + flattened_coeffs_plus2_str[:] + flattened_coeffs_minus2_str[:]
    
coefficients_plus_1 = Array.create('coeff_plus_1', shape=(num_basis_funcs_1, 2 * m1 + 1), vartype='BINARY')
coefficients_minus_1 = Array.create('coeff_minus_1', shape=(num_basis_funcs_1, 2 * m1 + 1), vartype='BINARY')
coefficients_plus_2 = Array.create('coeff_plus_2', shape=(num_basis_funcs_2, 2 * m2 + 1), vartype='BINARY')
coefficients_minus_2 = Array.create('coeff_minus_2', shape=(num_basis_funcs_2, 2 * m2 + 1), vartype='BINARY')

# Convert to numpy arrays and concatenate
coefficients_plus_1_np = np.array(coefficients_plus_1)
coefficients_minus_1_np = np.array(coefficients_minus_1)

coefficients_plus_2_np = np.array(coefficients_plus_2)
coefficients_minus_2_np = np.array(coefficients_minus_2)

pyqubo_coeffs_np = np.concatenate((coefficients_plus_1_np, coefficients_minus_1_np, coefficients_plus_2_np, coefficients_minus_2_np), axis=None)

# Create a mapping from sympy variable names to pyqubo variables
sympy_to_pyqubo_map = {}
num_minus = 0
num_plus = 0
index = 0
for var in enumerate(sorted(Solving_for_vars)):
    index += 1
    # print(var)
    var = str(var[1])
    parts = str(var).split('_')
    prefix = f"{parts[0]}_{parts[2]}"
    i = int(parts[1])
    j = int(parts[3])
    if prefix == 'P1_plus':
        sympy_to_pyqubo_map[f"{var}"] = coefficients_plus_1[i, j]
    elif prefix == 'P2_plus':
        sympy_to_pyqubo_map[f"{var}"] = coefficients_plus_2[i, j]
    elif prefix == 'P1_minus':
        sympy_to_pyqubo_map[f"{var}"] = coefficients_minus_1[i, j]
    elif prefix == 'P2_minus':
        sympy_to_pyqubo_map[f"{var}"] = coefficients_minus_2[i, j]

        
pyqubo_sse = convert_sympy_to_pyqubo(mse_with_penalty_str, sympy_to_pyqubo_map)
# Compile QUBO model
model = pyqubo_sse.compile()
qubo, offset = model.to_qubo()
bqm = BinaryQuadraticModel.from_qubo(qubo)
chain_strength = uniform_torque_compensation(bqm)

end_time_quantum_part1 = time.perf_counter()

time_quantum_part1 = end_time_quantum_part1 - start_time_quantum_part1
print(f"Time spent on quantum part 1: {time_quantum_part1}")  # will print the elapsed time in seconds

print(mse_with_penalty_str)

time_simulated_part2, time_simulated_part3, total_time_simulated_optimization, simulated_train_accuracy, simulated_train_precision, simulated_train_recall, simulated_train_f1, simulated_test_accuracy, simulated_test_precision, simulated_test_recall, simulated_test_f1 = run_annealing("SA", simulated_sampler, qubo, chain_strength, time_quantum_part1, num_basis_funcs_1, num_basis_funcs_2, m1, m2, degree1, degree2, x_data_train, y_data_train, z_data_train, x_data_test, y_data_test, z_data_test, dataset, num_reads=3000)