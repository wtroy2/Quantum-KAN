# Experiments  
This folder contains the experiments run from the paper. Note that due to quantum noise for some experiments your accuracy metrics may  
differ slightly and be better or worse for different runs.   

## Requirements  
To run experiments a key to use a D-Wave quantum annealer is needed. Keys can be obtained from D-Wave after making an account with them   
at https://cloud.dwavesys.com/leap/login/?next=/leap/.   

Once a key is obtained the experiments in this folder can be run by creating a file called dwave.env which each ipynb file reads from.   
This file is one line and contains you key and should look like the line below:  
DWAVE_API_TOKEN=DEV-**************************  

Where you replace DEV-************************** with your key. 
