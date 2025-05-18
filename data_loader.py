import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data_source = "C:\Users\Tanishq\Documents\resume\uni padebon application\project\data\cora.tgz\cora"
# Label mapping
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


def load_cora_content(file_path=r"C:\Users\Tanishq\Documents\resume\uni padebon application\project\scientific_paper_subject_prediction_task\cora.content"):
    """
    Loads the Cora content file and returns features, labels, paper IDs, and raw DataFrame.

    Args:
        file_path (str): Path to cora.content (default: '/content/cora/cora.content')

    Returns:
        X (np.ndarray): Feature matrix (2708 x 1433)
        y (np.ndarray): Encoded label vector (2708,)
        paper_ids (list of str): Original paper IDs
        label_to_index (dict): Manual mapping from string label to int
        index_to_label (dict): Reverse mapping
        df (pd.DataFrame): Raw dataframe for inspection
    """
    df = pd.read_csv(file_path, sep='\t', header=None)
    
    paper_ids = df.iloc[:, 0].values.tolist()
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    
    raw_labels = df.iloc[:, -1].values
    y = np.array([label_to_index[label] for label in raw_labels])

    return X, y, paper_ids, label_to_index, index_to_label, df

X, y, paper_ids, label_to_index, index_to_label, df = load_cora_content()

# Show sample of data
#df.head()
#df.iloc[:, -1].value_counts().plot(kind='bar', title='Class Distribution')
df.iloc[:, -1].value_counts().plot(kind='bar', title='Class Distribution')
plt.xlabel("Class Label")
plt.ylabel("Number of Papers")
plt.tight_layout()
plt.show()