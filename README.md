# Scientific Paper Subject Classification

A machine learning pipeline to predict the subjects of scientific papers from the Cora dataset using both content and citation information.

---

## Contents

1. `data_loader.py` – Loads and preprocesses the Cora dataset  
2. `gcn.py` – Trains a Graph Convolutional Network (GCN)  
3. `logistic_regression.py` – Trains a Logistic Regression baseline model (without citation features)  
4. `gcn_hyperparameter_tuning.py` – Performs hyperparameter search for the GCN  

---

## Requirements

### Built-in Modules
- `argparse`
- `itertools`

### Libraries
```txt
numpy==1.24.3
pandas==2.2.2
python==3.11.0
scikit-learn==1.5.2
torch==2.0.0+cu117
torch-geometric==2.6.1
```
---
### Run
1. Clone this repository
2. Ensure both cora.content and cora.cite exist in the same directory
3. Run the following in the command line
```
python gcn.py --content path/to/cora.content --cites path/to/cora.cites
```
4. Optional:
   Test the baseline model with:
   ```
   python logistic_baseline.py --content path/to/cora.content
   ```
   Test the hyperparameter grid search script with:

   ```
    gcn_hyperparameter_tuning.py --content path/to/cora.content --cites path/to/cora.cites
   ```
---

##### Approach 
This project benchmarks a Graph Convolutional Network (GCN) against a simple Logistic Regression model using 10 fold cross validation. 
The GCN approach is benchmarked upon a simple logistic regression model without the citation network features. To that end, an initial accuracy of 77.58%, 
shows an ~8% improvement with the GCN that includes the citation network features. 
The final accuracy of 83.94% is based on a hyperparameter grid search that automatizes various permutations of the hyperparameters, specifically the hidden layers, learning rate and optimizers. 
The hyperparameter grid search resulted in the best configuration of (16, 0.005 and Adam) respectively. The overall pipeline suggests that citation network features improve the prediction task.
