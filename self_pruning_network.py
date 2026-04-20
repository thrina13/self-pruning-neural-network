import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))  # ✅ FIXED
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    def gate_values(self):
        return torch.sigmoid(self.gate_scores)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    
    def sparsity_loss(self):
        loss = 0
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            gates = layer.gate_values()
            loss += (gates * (1 - gates)).mean()
        return loss

    def all_gates(self):
        return torch.cat([
            self.fc1.gate_values().view(-1),
            self.fc2.gate_values().view(-1),
            self.fc3.gate_values().view(-1),
            self.fc4.gate_values().view(-1),
        ])



transform = transforms.Compose([transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128)



def train_model(lam):
    model = Net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10   # fast + enough

    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)

            ce = F.cross_entropy(out, y)
            sp = model.sparsity_loss()

            # 🔥 Balanced loss (important)
            loss = ce + lam * 100 * sp

            loss.backward()
            optimizer.step()

    # Accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total

   
    gates = model.all_gates()
    sparsity = (gates < 0.1).sum().item() / gates.numel() * 100

    return model, acc, sparsity



lambdas = [1e-5, 1e-4, 1e-3]

results = []
best_model = None
best_acc = 0
best_lam = None

for lam in lambdas:
    print(f"\nRunning λ = {lam}")
    model, acc, sp = train_model(lam)
    print(f"Accuracy: {acc:.2f}% | Sparsity: {sp:.2f}%")

    results.append([lam, acc, sp])

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_lam = lam



with open("results_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Lambda", "Accuracy (%)", "Sparsity (%)"])
    writer.writerows(results)



gates = best_model.all_gates().detach().cpu().numpy()

plt.hist(gates, bins=50)
plt.title(f"Gate Distribution (λ={best_lam})")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.savefig("gate_distribution.png")
plt.show()

print("\nDONE ✅ Files generated")
