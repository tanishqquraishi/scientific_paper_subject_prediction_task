import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from itertools import product
from data_loader import load_cora_content
from gcn import load_edge_index

# Load data
X, y, paper_ids, label_to_index, index_to_label, df = load_cora_content("cora.content")
edge_index = load_edge_index("cora.cites", paper_ids)
features = torch.tensor(X, dtype=torch.float)
labels = torch.tensor(y, dtype=torch.long)

# Define the GCN model
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

def train_gcn(hidden_dim, lr, optimizer_name, weight_decay=5e-4, epochs=200):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_preds = torch.zeros(len(labels), dtype=torch.long)

    for train_idx, test_idx in skf.split(features, labels):
        model = GCN(input_dim=features.shape[1], hidden_dim=hidden_dim, output_dim=len(label_to_index))

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        for epoch in range(epochs):
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

    acc = accuracy_score(labels.numpy(), all_preds.numpy())
    return acc

# Hyperparameter grid
hidden_dims = [16, 32, 64]
learning_rates = [0.01, 0.005]
optimizers = ['adam', 'sgd']

# Run grid search
best_acc = 0
best_config = None
results = []

for hidden_dim, lr, opt in product(hidden_dims, learning_rates, optimizers):
    print(f"Training: hidden_dim={hidden_dim}, lr={lr}, optimizer={opt}")
    acc = train_gcn(hidden_dim, lr, opt)
    results.append((hidden_dim, lr, opt, acc))
    if acc > best_acc:
        best_acc = acc
        best_config = (hidden_dim, lr, opt)

print("\n✅ Best Accuracy:", round(best_acc * 100, 2), "%")
print("🧪 Best Config:", best_config)

print("\n📋 All Results:")
for r in sorted(results, key=lambda x: -x[3]):
    print(f"Hidden={r[0]:<3} | LR={r[1]:<6} | Opt={r[2]:<4} | Acc={r[3]*100:.2f}%")