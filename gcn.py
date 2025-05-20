import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from data_loader import load_cora_content

# Argument parser for command line running
parser = argparse.ArgumentParser(description="Train GCN on Cora dataset.")
parser.add_argument('--content', type=str, required=True, help="Path to cora.content file")
parser.add_argument('--cites', type=str, required=True, help="Path to cora.cites file")
args = parser.parse_args()

# Load data
X, y, paper_ids, label_to_index, index_to_label, df = load_cora_content(args.content)

# Build citation graph
def load_edge_index(cites_path, paper_ids):
    paper_id_to_idx = {str(pid): idx for idx, pid in enumerate(paper_ids)}  
    edge_list = []
    missing = 0

    with open(cites_path, 'r') as f:
        for line in f:
            cited, citing = line.strip().split()
            cited = str(cited)
            citing = str(citing)
            if cited in paper_id_to_idx and citing in paper_id_to_idx:
                edge_list.append([paper_id_to_idx[citing], paper_id_to_idx[cited]])
            else:
                missing += 1

    print(f"Loaded {len(edge_list)} citation edges, skipped {missing} due to ID mismatch.")
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

edge_index = load_edge_index(args.cites, paper_ids)
features = torch.tensor(X, dtype=torch.float)
labels = torch.tensor(y, dtype=torch.long)

# Define GCN  parameters
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 10-fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_preds = torch.zeros(len(labels), dtype=torch.long)

for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
    model = GCN(input_dim=features.shape[1], hidden_dim=16, output_dim=len(label_to_index))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = F.cross_entropy(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(features, edge_index)
        preds = out[test_idx].argmax(dim=1)
        all_preds[test_idx] = preds

# Save predictions
with open("gcn_predictions.tsv", "w") as f:
    for pid, pred in zip(paper_ids, all_preds):
        class_label = index_to_label[pred.item()]
        f.write(f"{pid}\t{class_label}\n")

accuracy = accuracy_score(labels.numpy(), all_preds.numpy())
print(f"GCN Accuracy: {accuracy * 100:.2f}%")