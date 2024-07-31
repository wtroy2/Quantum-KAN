import quantum_kan 
import symengine as se
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import mean_squared_error, r2_score
from pyqubo import Binary, Array
import os
from dwave.embedding.chain_breaks import majority_vote
from dimod import BinaryQuadraticModel
import dwave.inspector
# Solve the QUBO using a quantum annealer
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler, FixedEmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
import time 
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from qkan import KAN
import torch.nn as nn
import minorminer
import networkx as nx

# Set your D-Wave API token
import os
# env_vars = !cat dwave.env
# for var in env_vars:
#     key, value = var.split('=')
#     os.environ[str(key)] = str(value)
# #Advantage_system6.4 or Advantage2_prototype2.3
# # sampler = EmbeddingComposite(DWaveSampler(solver='Advantage2_prototype2.3'))
# sampler = LeapHybridSampler(solver="hybrid_binary_quadratic_model_version2")

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
            if ('P' in part) or ('AUX' in part):
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
    
def separate_vars(solution):
    """
    Separate coefficients and auxiliary variables from the solution dictionary.

    :param solution: The solution dictionary containing both coefficients and auxiliary variables.
    :return: Two dictionaries, one for coefficients and one for auxiliary variables.
    """
    aux_vars = {k: v for k, v in solution.items() if k.startswith('aux')}
    coeff_vars = {k: v for k, v in solution.items() if k.startswith('coeff')}
    
    return aux_vars, coeff_vars

def create_reverse_lookup_dict(original_dict):
    """
    Create a reverse lookup dictionary from the given dictionary.

    :param original_dict: The original dictionary to reverse.
    :return: A new dictionary where keys are the original values and values are the original keys.
    """
    reverse_dict = {v: k for k, v in original_dict.items()}
    return reverse_dict

def map_solution_variables(solution, reverse_dict):
    """
    Map solution variables from PyQUBO to SymPy using the reverse dictionary.

    :param solution: The solution dictionary with PyQUBO variables.
    :param reverse_dict: The reverse lookup dictionary to map PyQUBO variables to SymPy variables.
    :return: A new dictionary with SymPy variables as keys and solution values.
    """
    mapped_solution = {}
    for pyqubo_var, value in solution.items():
        sympy_var = reverse_dict.get(Binary(pyqubo_var))
        if sympy_var is not None:
            mapped_solution[sympy_var] = value
        else:
            print(f"Warning: PyQUBO variable {pyqubo_var} not found in reverse mapping.")
    return mapped_solution

def separate_aux_and_binary_vars(mapped_solution, include_aux_in_binary=False):
    """
    Separate coefficients and auxiliary variables from the mapped solution dictionary.

    :param mapped_solution: The mapped solution dictionary containing both coefficients and auxiliary variables.
    :return: Two dictionaries, one for coefficients and one for auxiliary variables.
    """
    aux_vars = {str(k).replace('"','').replace("'",""): v for k, v in mapped_solution.items() if str(k).replace('"','').replace("'","").startswith('AUX')}
    if include_aux_in_binary == True:
        binary_vars = {str(k).replace('"','').replace("'",""): v for k, v in mapped_solution.items() if  str(k).replace('"','').replace("'","")}
    else:
        binary_vars = {str(k).replace('"','').replace("'",""): v for k, v in mapped_solution.items() if not str(k).replace('"','').replace("'","").startswith('AUX')}
    
    return aux_vars, binary_vars

def check_aux_variables(binary_vars, aux_vars, aux_dict_all):
    """
    Check if auxiliary variables correctly represent the product of their respective binary variables.

    :param binary_vars: Dictionary of binary variable values (e.g., {'x1': 0, 'x2': 1, ...})
    :param aux_vars: Dictionary of auxiliary variable values (e.g., {'z1': 1, 'z2': 0, ...})
    :param aux_dict_all: Dictionary mapping auxiliary variables to their respective binary variable products
                         (e.g., {'z1': x1*x2, 'z2': x2*x3, ...})
    :return: Dictionary indicating whether each auxiliary variable is correct (e.g., {'z1': True, 'z2': False, ...})
    """
    results = {}

    for aux_var, aux_expr in aux_dict_all.items():
        # Extract the binary variables involved in the product
        vars_in_expr = aux_expr.split('*')
        product_value = 1
        aux_var = str(aux_var)

        # Calculate the expected value by multiplying the values of the binary variables
        for var in vars_in_expr:
            var = str(var)
            if var in binary_vars:
                product_value *= binary_vars[var]
            else:
                raise ValueError(f"Variable {var} not found in binary_vars dictionary")

        # Check if the auxiliary variable's value matches the expected product value
        if aux_var in aux_vars:
            results[aux_var] = (aux_vars[aux_var] == product_value)
        else:
            raise ValueError(f"Auxiliary variable {aux_var} not found in aux_vars dictionary")

    return results
    
def create_dataset(f, 
                   n_var=2, 
                   ranges = [0,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=True,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [0,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)
        
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:, i] = torch.rand(train_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
        test_input[:, i] = torch.rand(test_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
        
    train_label = f(train_input)
    test_label = f(test_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = (train_label - mean_label) / std_label
        test_label = (test_label - mean_label) / std_label

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)
    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset

def split_dataset(dataset, splits):
    """
    Split dataset into multiple smaller datasets.
    
    Args:
    -----
        dataset : dict
            The dataset containing 'train_input' and 'train_label'.
        splits : list
            List of integers specifying the size of each split.
            
    Returns:
    --------
        split_datasets : list
            List of datasets, each containing 'train_input' and 'train_label'.
    """
    total_size = sum(splits)
    assert total_size == len(dataset['train_input']), "Sum of splits must equal the total dataset size."
    
    # Create a list of indices and shuffle them
    indices = torch.randperm(total_size)
    
    split_datasets = []
    start_idx = 0
    
    for split_size in splits:
        end_idx = start_idx + split_size
        split_indices = indices[start_idx:end_idx]
        
        split_input = dataset['train_input'][split_indices]
        split_label = dataset['train_label'][split_indices]
        
        split_dataset = {
            'train_input': split_input,
            'train_label': split_label,
            'test_input': torch.empty(0, split_input.shape[1]),  # Empty tensor with the same number of columns as train_input
            'test_label': torch.empty(0, split_label.shape[1])   # Empty tensor with the same number of columns as train_label
        }
        
        split_datasets.append(split_dataset)
        start_idx = end_idx
        
    return split_datasets


def train_classical_optimizer(dataset, degrees, optimizer, steps, learning_rates, T, U, Z_ideal, Z_ideal_flat):
    best_lr = None
    best_mse = 100000
    grid_points = np.c_[T.ravel(), U.ravel()]
    grid_points_tensor = torch.from_numpy(grid_points).float()

    for lr in learning_rates:
        model = KAN(width=[2, 1, 1], degrees=degrees, bias_trainable=False, base_fun=ZeroBaseFunction()).float()
        start = time.perf_counter()
        results = model.train(dataset, opt=optimizer, steps=steps, log=10, lr=lr)
        end = time.perf_counter()
        training_time = end - start
        print(f"Training time in seconds of lr={lr}: {training_time}")  # will print the elapsed time in seconds

        # Compute the model predictions
        with torch.no_grad():
            Z_model = model(grid_points_tensor).numpy().reshape(T.shape)
            
        # Flatten the arrays for comparison
        Z_model_flat = Z_model.ravel()

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(Z_ideal_flat, Z_model_flat)
        r2 = r2_score(Z_ideal_flat, Z_model_flat)
        if mse < best_mse:
            best_mse = mse
            best_lr = lr
        print(f"Learning Rate: {lr}, Train MSE: {mse}, Train r2: {r2}")

    print(f"Best Learning Rate: {best_lr} with Train MSE: {best_mse}")

    # Plot the results using the best learning rate
    model = KAN(width=[2, 1, 1], degrees=degrees, bias_trainable=False, base_fun=ZeroBaseFunction()).float()


    start = time.perf_counter()

    results = model.train(dataset, opt=optimizer, steps=steps, log=10, lr=best_lr)
    end = time.perf_counter()
    time_best_lr = end - start

    print(f"Training time in seconds: {time_best_lr}")  # will print the elapsed time in seconds

    # Print final Bezier coefficients for each layer
    for layer_idx, layer in enumerate(model.act_fun):
        print(f"Layer {layer_idx} Bezier coefficients:")
        print(layer.coef.data.cpu().numpy())

    # Compute the model predictions
    with torch.no_grad():
        Z_model = model(grid_points_tensor).numpy().reshape(T.shape)

    # Plot the ideal function
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.contourf(T, U, Z_ideal, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title('Ideal Function')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot the model predictions
    plt.subplot(1, 2, 2)
    plt.contourf(T, U, Z_model, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title('Model Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.tight_layout()
    plt.show()

    # Calculate metrics
    # Flatten the arrays for comparison
    Z_model_flat = Z_model.ravel()

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(Z_ideal_flat, Z_model_flat)

    # Calculate R-squared (R²)
    r2 = r2_score(Z_ideal_flat, Z_model_flat)

    print(f"Training time: {time_best_lr}, mse: {mse}, r2: {r2}")

    return time_best_lr, mse, r2

def run_annealing(type, sampler, problem, chain_strength, time_part1, num_basis_funcs_1, num_basis_funcs_2, num_basis_funcs_3, m1, m2, m3, num_aux, sympy_to_pyqubo_map, aux_dict_str, bias_coefficient, T, U, Z_ideal, Z_ideal_flat, num_reads=1000, annealing_time=20):
    #Check if quantum annealer, hybrid quantum annealer, or simulated annealer
    if type == "QA":
        response = sampler.sample(problem, num_reads=num_reads, chain_strength=chain_strength, annealing_time=annealing_time)
        # Extract and print timing information
        timing_info = response.info['timing']

        print("Timing Information:")
        for key, value in timing_info.items():
            print(f"{key}: {value} microseconds")

        # Specific time spent on the quantum processor
        qpu_access_time = timing_info.get('qpu_access_time', 'N/A')
        print(f"\nQPU Access Time: {qpu_access_time} microseconds")

        time_part2 = qpu_access_time * (10**(-6))
        print(f"Time spent on quantum annealer part 2: {time_part2}")  # will print the elapsed time in seconds

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
        coeff_value_1 = sum(2**j * solution[f'coeff_plus_1[{i}][{j + m1 - 1}]'] for j in range(-m1 + 1, 1))
        optimized_coefficients_1.append(coeff_value_1)
        
    optimized_coefficients_2 = []
    for i in range(num_basis_funcs_2):
        coeff_value_2 = sum(2**j * solution[f'coeff_plus_2[{i}][{j + m2 - 1}]'] for j in range(-m2 + 1, 1))
        optimized_coefficients_2.append(coeff_value_2)
        
    optimized_coefficients_3 = []
    for i in range(num_basis_funcs_3):
        coeff_value_3 = sum(2**j * solution[f'coeff_plus_3[{i}][{j + m3 - 1}]'] for j in range(-m3 + 1, 1))
        optimized_coefficients_3.append(coeff_value_3)
        
    optimized_aux = []
    for i in range(num_aux):
        coeff_value_1 = solution[f'aux[{i}]']
        optimized_aux.append(coeff_value_1)
    end_time_part3 = time.perf_counter()

    time_part3 = end_time_part3 - start_time_part3
    print(f"Time spent on part 3: {time_part3}")  # will print the elapsed time in seconds

    total_time_optimization = time_part1 + time_part2 + time_part3
    print(f"Time spent on optimization: {total_time_optimization}")  # will print the elapsed time in seconds
    print(f"optimized_coefficients_1: {optimized_coefficients_1}")
    print(f"optimized_coefficients_2: {optimized_coefficients_2}")
    print(f"optimized_coefficients_3: {optimized_coefficients_3}")

    # Separate coefficients and auxiliary variables
    aux_vars, coeff_vars = separate_vars(solution)
    # Create reverse lookup dictionary
    reverse_dict = create_reverse_lookup_dict(sympy_to_pyqubo_map)
    # Map solution variables from PyQUBO to SymPy
    mapped_solution = map_solution_variables(solution, reverse_dict)
    include_aux_in_binary = True
    aux_vars, binary_vars = separate_aux_and_binary_vars(mapped_solution, include_aux_in_binary)
    # Check the auxiliary variable values
    results_aux_check = check_aux_variables(binary_vars, aux_vars, aux_dict_str)

    # Print the results
    for aux_var, is_correct in results_aux_check.items():
        print(f"{aux_var}: {'Correct' if is_correct else 'Incorrect'}")

    # Compute Bézier curve values for each grid point
    B1_values = np.array([[bezier_curve(optimized_coefficients_1, t) + bias_coefficient * t**2 for t in t_row] for t_row in T])
    B2_values = np.array([[bezier_curve(optimized_coefficients_2, u) + bias_coefficient * u**2 for u in u_row] for u_row in U])


    # Compute Bézier3 using the output of Bézier1 and Bézier2
    Z_model = np.array([[bezier_curve(optimized_coefficients_3, (B1_values[i, j] + B2_values[i, j])) + bias_coefficient * (B1_values[i, j] + B2_values[i, j])**2 for j in range(len(T))] for i in range(len(U))])

    # Plotting the 3D surface plots side by side
    fig = plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.contourf(T, U, Z_ideal, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title('Ideal Function')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot the model predictions
    plt.subplot(1, 2, 2)
    plt.contourf(T, U, Z_model, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title('Quantum Annealing Model Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Calculate metrics
    # Flatten the arrays for comparison
    Z_model_flat = Z_model.ravel()

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(Z_ideal_flat, Z_model_flat)

    # Calculate R-squared (R²)
    r2 = r2_score(Z_ideal_flat, Z_model_flat)

    print(f"Training time: {total_time_optimization}, MSE: {mse:.4f}, r2: {r2:.4f}")
    print(f"time_part1: {time_part1}, time_part2: {time_part2}, time_part3: {time_part3}")
    return time_part2, time_part3, total_time_optimization, mse, r2

# Defining variables for the networks
degree1 = 1
degree2 = 1
degree3 = 1
degrees=[degree1,degree3]
m1 = 1
m2 = 3
m3 = 3
penalty_multiplier = 25
bias_coefficient = 0.0
is_fractional=True

num_basis_funcs_1 = degree1 + 1
num_basis_funcs_2 = degree2 + 1
num_basis_funcs_3 = degree3 + 1

save_file1 = "35_18_mse_save1.json"
save_file2 = "35_18_mse_save2.json"
save_file3 = "35_18_mse_save3.json"
save_file4 = "35_18_mse_save4.json"

# Define variables to keep track of time
total_time_quantum = []
total_time_simulated = []
total_time_adam = []
total_time_sgd = []
total_time_adagrad = []

dataset_multiplier = 3
dataset_size = 2400000
# Create dataset with normalized inputs between 0 and 1
f = lambda x: x[:, [0]] / (np.exp(x[:, [1]]) + (np.exp( -1 * x[:, [1]]))) * dataset_multiplier
dataset = create_dataset(f, n_var=2, train_num=dataset_size, test_num=0, ranges=[0, 1])

# Define the sizes of the splits
splits = [2000000, 100000, 100000, 100000, 100000]

# Split the dataset
split_datasets = split_dataset(dataset, splits)

# Example usage
for i, split_dataset in enumerate(split_datasets):
    print(f"Dataset {i+1}:")
    print(f"  Input shape: {split_dataset['train_input'].shape}")
    print(f"  Label shape: {split_dataset['train_label'].shape}")
    if i == 0:
        dataset = split_dataset
    elif i == 1:
        dataset_split1 = split_dataset
    elif i == 2:
        dataset_split2 = split_dataset
    elif i == 3:
        dataset_split3 = split_dataset
    elif i == 4:
        dataset_split4 = split_dataset
    
X = dataset['train_input']
y = dataset['train_label']
x_data = X[:, 0].numpy()
y_data = X[:, 1].numpy()
z_data = y[:, 0].numpy()

# Generate a grid of t values for the 3D plot
t_values = np.linspace(0, 1, 30)
u_values = np.linspace(0, 1, 30)
T, U = np.meshgrid(t_values, u_values)

# Compute the ideal function values
ideal_func = lambda t, u: (t/ (np.exp(u) + np.exp(-1 * u))) * dataset_multiplier
Z_ideal = np.array([[ideal_func(t, u) for t, u in zip(t_row, u_row)] for t_row, u_row in zip(T, U)])

Z_ideal_flat = Z_ideal.ravel()


start_time_quantum_part1 = time.perf_counter()
# Call the C++ function and unpack the returned values
result = quantum_kan.compute_mse_with_penalty(
    d1=degree1, d2=degree2, d3=degree3,
    m1=m1, m2=m2, m3=m3,
    penalty_multiplier=penalty_multiplier,
    bias_coefficient=bias_coefficient,
    is_fractional=is_fractional,
    x_data=x_data.tolist(),
    y_data=y_data.tolist(),
    z_data=z_data.tolist(),
    save_filename=save_file1
)

mse_with_penalty_str, aux_dict_str, coeffs_plus1_str, coeffs_plus2_str, coeffs_plus3_str = result

flattened_coeffs_plus1_str = [item for sublist in coeffs_plus1_str for item in sublist]
flattened_coeffs_plus2_str = [item for sublist in coeffs_plus2_str for item in sublist]
flattened_coeffs_plus3_str = [item for sublist in coeffs_plus3_str for item in sublist]

# Define P variables (flatten the matrix of Binary variables to a list)
Solving_for_vars = flattened_coeffs_plus1_str[:] + flattened_coeffs_plus2_str[:] + flattened_coeffs_plus3_str[:]

for aux_var, aux_expr in aux_dict_str.items():
    Solving_for_vars.append(aux_var)
    
coefficients_plus_1 = Array.create('coeff_plus_1', shape=(num_basis_funcs_1, m1), vartype='BINARY')
coefficients_plus_2 = Array.create('coeff_plus_2', shape=(num_basis_funcs_2, m2), vartype='BINARY')
coefficients_plus_3 = Array.create('coeff_plus_3', shape=(num_basis_funcs_3, m3), vartype='BINARY')
num_aux = len(aux_dict_str)
coefficients_aux = Array.create('aux', shape=(num_aux,), vartype='BINARY')

# Convert to numpy arrays and concatenate
coefficients_plus_1_np = np.array(coefficients_plus_1)
# coefficients_minus_1_np = np.array(coefficients_minus_1)

coefficients_plus_2_np = np.array(coefficients_plus_2)
# coefficients_minus_2_np = np.array(coefficients_minus_2)
coefficients_plus_3_np = np.array(coefficients_plus_3)

coefficients_aux_np = np.array(coefficients_aux)

pyqubo_coeffs_np = np.concatenate((coefficients_plus_1_np, coefficients_plus_2_np, coefficients_plus_3_np, coefficients_aux_np), axis=None)

# Create a mapping from sympy variable names to pyqubo variables
sympy_to_pyqubo_map = {}
num_minus = 0
num_plus = 0
index = 0
# Mapping for P variables
for var in sorted(Solving_for_vars):
    var = str(var)
    if '_' in var:
        parts = var.split('_')
        prefix = f"{parts[0]}_{parts[2]}"
        i = int(parts[1])
        j = int(parts[3])
        if prefix == 'P1_plus':
            sympy_to_pyqubo_map[f"{var}"] = coefficients_plus_1[i, j]
        elif prefix == 'P2_plus':
            sympy_to_pyqubo_map[f"{var}"] = coefficients_plus_2[i, j]
        elif prefix == 'P3_plus':
            sympy_to_pyqubo_map[f"{var}"] = coefficients_plus_3[i, j]

# Mapping for AUX variables
aux_var_to_index = {var: idx for idx, var in enumerate(aux_dict_str.keys())}
for aux_var in aux_dict_str.keys():
    sympy_to_pyqubo_map[f"{aux_var}"] = coefficients_aux[aux_var_to_index[aux_var]]
        
pyqubo_sse = convert_sympy_to_pyqubo(mse_with_penalty_str, sympy_to_pyqubo_map)
# Compile QUBO model
model = pyqubo_sse.compile()
qubo, offset = model.to_qubo()
bqm = BinaryQuadraticModel.from_qubo(qubo)
chain_strength = uniform_torque_compensation(bqm) #* 2.0

end_time_quantum_part1 = time.perf_counter()
    
    
time_quantum_part1 = end_time_quantum_part1 - start_time_quantum_part1
print(f"Time spent on quantum part 1: {time_quantum_part1}")  # will print the elapsed time in seconds

simulated_time_part2, simulated_time_part3, simulated_total_time_optimization, simulated_mse, simulated_r2 = run_annealing("SA", simulated_sampler, qubo, chain_strength, time_quantum_part1, num_basis_funcs_1, num_basis_funcs_2, num_basis_funcs_3, m1, m2, m3, num_aux, sympy_to_pyqubo_map, aux_dict_str, bias_coefficient, T, U, Z_ideal, Z_ideal_flat, num_reads=10000)