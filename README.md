# Quantum-KAN
The repository contains 2 packages. The first is the quantum_kan package which is implimented in c++ bound to python and the second in qkan which is a modified version of the original pykan package. All experiments shown in the figures from the paper 'Sparks of Quantum Advantage and Rapid Retraining in Machine Learning' are saved in the jupyter notebooks in the experiments folder of this repo.

## quantum_kan
quantum_kan creates initial problem setup and the forward pass for the Quantum Bezier-KAN. 
Prerequisites:  
1. Install symengine
```bash
conda install symengine -c conda-forge
```

2. Install Eigen. On ubuntu this can be done with:  
```bash
apt-get install libeigen3-dev
```

3. Install OpenBLAS. On ubuntu this can be done with:  
```bash
apt-get install libopenblas-dev
```

It is found in the base directory of the repo and can be installed into python using pip:
```bash
pip install .
```
It currently has only 2 functions. 
1. 'compute_mse_with_penalty' which is used to create a 2 layer Quantum Bezier-KAN and put it through a forward pass.  
Inputs for the function (all variables are required except for load_filename and save_filename):  
d1: degree of the first bottom layer Bezier function  
d2: degree of the second bottom layer Bezier function  
d3: degree of the top layer Bezier function  
d1: number of qubits used to represent each control point in the first bottom layer Bezier function  
d2: number of qubits used to represent each control point in the second bottom layer Bezier function  
d3: number of qubits used to represent each control point in the top layer Bezier function  
penalty_multiplier: Multiplier for the penalty function. This is to enforce the problem constraints and ranges from 15-25 in the paper.  
bias_coefficient: This is an experimental feature and should just be set to 0 for now.  
is_fractional: Currently the 2 layer KAN only supports positive integer coefficients (false) or fractional integer coefficients (true).  
x_data: Input data for x  
y_data: Input data for y  
z_data: Input data for z  
load_filename: If loading the network from a save file please provide the name of this file to load.  
save_filename: If saving the network to a save file please provide the name of the file you want to save it to.  

2. 'compute_mse_with_penalty_categorical' which is used to create a 1 layer Quantum Bezier-KAN and put it through a forward pass.  
Inputs for the function (all variables are required except for load_filename and save_filename):  
d1: degree of the first bottom layer Bezier function  
d2: degree of the second bottom layer Bezier function  
d1: 4*d1 + 2 is the number of qubits used to represent each control point in the first bottom layer Bezier function  
d2: 4*d2 + 2 is the number of qubits used to represent each control point in the second bottom layer Bezier function  
penalty_multiplier: Multiplier for the penalty function. This is to enforce the problem constraints and ranges from 15-25 in the paper.  
bias_coefficient: This is an experimental feature and should just be set to 0 for now.  
x_data_train: Input training data for x  
y_data_train: Input training data for y  
z_data_train: Input training data for z  
x_data_test: Input testing data for x  
y_data_test: Input testing data for y  
z_data_test: Input testing data for z  
test_multiplier: Float value for the test data weighting.  
load_filename: If loading the network from a save file please provide the name of this file to load.  
save_filename: If saving the network to a save file please provide the name of the file you want to save it to.  


## qkan
qkan is a modified version of the original pykan package. qkan does NOT involve anything quantum. It is just for testing speed against quantum_kan. 
This package can be found in the folder labeled 'modified_original_kan' and does minimal modifications to use the Bezier function instead of b-splines 
and allow for different degrees of the Bezier function on each layer. All credit for the original pykan implimentation goes to Ziming Liu and colleagues.  
Their repo for pykan can be found here: https://github.com/KindXiaoming/pykan.   

## NOTES
* This repository in its current state is not production level and is built for scientific testing purposes.
* There are future plans to speed up the forward pass before quantum optimization and update the repo and paper. 
* If you have any questions feel free to reach out to William Troy at troywilliame@gmail.com

## Citation
Troy, W. Sparks of Quantum Advantage and Rapid Retraining in Machine Learning. Preprint at http://arxiv.org/abs/2407.16020 (2024).  
