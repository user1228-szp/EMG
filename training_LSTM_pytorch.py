import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar datos desde CSV
def load_data():
    df = pd.read_csv('data.csv')
    x_data = df.iloc[:, :-1].values
    y_data = df.iloc[:, -1].values
    return train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Definir el modelo LSTM en PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, mem_cell_ct):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, mem_cell_ct, batch_first=True)
        self.fc = nn.Linear(mem_cell_ct, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Entrenar y evaluar el modelo LSTM en PyTorch
def train_lstm_torch():
    x_train, x_val, y_train, y_val = load_data()
    x_train, x_val = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_val, dtype=torch.float32)
    y_train, y_val = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    model = LSTMModel(x_train.shape[1], mem_cell_ct=100)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val)
        
        print(f"Epoch {epoch+1}: Train Loss: {loss.item():.3f}, Validation Loss: {val_loss.item():.3f}")

if __name__ == "__main__":
    train_lstm_torch()