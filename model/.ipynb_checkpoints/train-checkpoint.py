import torch
import torch.nn as nn
import numpy as np

class DragMLP(nn.Module):
    def __init__(self, in_features=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

def main():
    X = np.load("/Users/joshuahellewell/arm-project/model/X.npy")
    y = np.load("/Users/joshuahellewell/arm-project/model/y.npy")

    # normalize inputs -- important since length/width/height/angle
    # have very different scales
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    n = X_t.shape[0]
    n_train = int(n * 0.8)
    X_train, X_val = X_t[:n_train], X_t[n_train:]
    y_train, y_val = y_t[:n_train], y_t[n_train:]

    model = DragMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_epochs = 200
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val)
            print(f"epoch {epoch:4d}  train_loss {loss.item():.5f}  val_loss {val_loss.item():.5f}")

    torch.save(model.state_dict(), "/Users/joshuahellewell/arm-project/model/drag_mlp.pt")
    np.save("/Users/joshuahellewell/arm-project/model/X_mean.npy", X_mean)
    np.save("/Users/joshuahellewell/arm-project/model/X_std.npy", X_std)
    print("Saved model/drag_mlp.pt")

if __name__ == "__main__":
    main()