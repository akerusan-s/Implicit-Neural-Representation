import torch

from models.siren_nondefault_init import RktvModel



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------
# 1. Определение архитектуры свёрточной сети
# ----------------------------------------------
class Conv1DRegressor(nn.Module):
    """
    Одномерная свёрточная сеть для регрессии последовательностей.
    Вход:  (batch_size, 2, seq_len)   # 2 канала (двумерные переменные)
    Выход: (batch_size, 2, seq_len)   # предсказанный ряд той же размерности
    """
    def __init__(self, in_channels=2, out_channels=2, seq_len=None):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 4, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(4)
        self.relu1  = nn.ReLU()

        self.conv2 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(8)
        self.relu2  = nn.ReLU()

        # Выходной слой: свёртка 1x1 для изменения числа каналов
        self.conv_out = nn.Conv1d(8, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (batch, 2, seq_len)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv_out(x)               # (batch, out_channels, seq_len)
        return x

# ----------------------------------------------
# 2. Генерация синтетических данных для примера
# ----------------------------------------------
def generate_data(num_samples=1000, seq_len=50):
    """
    Создаёт случайные входные последовательности и целевые последовательности.
    Вход:  двумерный ряд (синусоиды с шумом)
    Цель:  сдвинутая версия входа (например, сглаженная или предсказание на 1 шаг вперёд)
    Здесь для простоты цель = вход + шум (модель учится восстанавливать исходный сигнал).
    """
    X = []
    y = []
    for _ in range(num_samples):
        t = torch.linspace(0, 4*torch.pi, seq_len)
        # Две переменные: синус и косинус с разными частотами + шум
        signal1 = torch.sin(0.5 * t) + 0.1 * torch.randn(seq_len)
        signal2 = torch.cos(1.2 * t) + 0.1 * torch.randn(seq_len)
        inp = torch.stack([signal1, signal2], dim=0)  # (2, seq_len)

        # Цель: например, та же последовательность, но с небольшим сглаживанием
        target1 = torch.sin(0.5 * t)  # чистый сигнал без шума
        target2 = torch.cos(1.2 * t)
        out = torch.stack([target1, target2], dim=0)   # (2, seq_len)

        X.append(inp)
        y.append(out)

    X = torch.stack(X)  # (num_samples, 2, seq_len)
    y = torch.stack(y)
    return X, y

# ----------------------------------------------
# 3. Подготовка данных и обучение
# ----------------------------------------------
# Параметры
batch_size = 1
seq_len = 50
epochs = 10
lr = 0.001

# Генерация данных
X, y = generate_data(num_samples=2000, seq_len=seq_len)

# Разделение на обучающую и валидационную выборки
split = int(0.8 * len(X))
train_dataset = TensorDataset(X[:split], y[:split])
val_dataset   = TensorDataset(X[split:], y[split:])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Conv1DRegressor(in_channels=2, out_channels=2)

# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# Цикл обучения
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0.0
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item() * batch_X.size(0)
#
#     train_loss /= len(train_loader.dataset)
#
#     # Валидация
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch_X, batch_y in val_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             val_loss += loss.item() * batch_X.size(0)
#
#     val_loss /= len(val_loader.dataset)
#
#     print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

model.eval()
with torch.no_grad():
    sample_X, sample_y = next(iter(val_loader))
    print(sample_X[0].shape)
    pred = model(sample_X).reshape((2, 50)).T
    print(f"\nПример предсказания: форма входа {sample_X.shape}, выхода {pred.shape}")