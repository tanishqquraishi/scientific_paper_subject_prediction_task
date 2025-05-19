import pandas as pd
import numpy as np

# Map Labels
label_to_index = {
    "Case_Based": 0,
    "Genetic_Algorithms": 1,
    "Neural_Networks": 2,
    "Probabilistic_Methods": 3,
    "Reinforcement_Learning": 4,
    "Rule_Learning": 5,
    "Theory": 6
}
index_to_label = {v: k for k, v in label_to_index.items()}

# Load Data
def load_cora_content(file_path):
    """
    Loads the Cora.content file.
    Args:
        file_path (str): Path to cora.content
    Returns:
        X (np.ndarray), y (np.ndarray), paper_ids (list), label dicts, DataFrame
    """
    #df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    df = pd.read_csv(file_path, sep=r'\s+', engine='python', header=None)
    paper_ids = df.iloc[:, 0].tolist()
    X = df.iloc[:, 1:-1].values.astype(np.float32) # input 
    raw_labels = df.iloc[:, -1].values
    y = np.array([label_to_index[label] for label in raw_labels]) # output
    return X, y, paper_ids, label_to_index, index_to_label, df