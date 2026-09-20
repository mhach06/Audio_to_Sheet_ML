import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import AudioDataset
from model import PianoTranscriptionCNN


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = AudioDataset("data")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = PianoTranscriptionCNN()

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss = nn.BCELoss()

    for epoch in range(50):
        # TRAINING PHASE
        model.train()
        for batch_idx, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss_value = loss(output, Y)
            loss_value.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/50], Batch [{batch_idx}], Loss: {loss_value.item():.4f}")

        # VALIDATION PHASE
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X = X.to(device)
                Y = Y.to(device)

                output = model(X)
                val_loss += loss(output, Y).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/50], Validation Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "piano_model.pth")

if __name__ == "__main__":
    main()