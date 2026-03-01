# predictor/ml_model/model.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ConvLSTMCell(nn.Module):
    """Ячейка ConvLSTM"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvLSTMCell, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.hidden_dim = hidden_dim

    def forward(self, x, hc):
        h, c = hc
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ArcticIcePredictor(nn.Module):
    """Модель для прогнозирования арктического льда (исправленная версия)"""

    def __init__(self, input_channels=7, hidden_dims=[16, 32, 16], grid_size=(30, 60)):
        super(ArcticIcePredictor, self).__init__()

        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.grid_size = grid_size

        # ConvLSTM слои
        self.convlstm1 = ConvLSTMCell(input_channels, hidden_dims[0])
        self.convlstm2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1])
        self.convlstm3 = ConvLSTMCell(hidden_dims[1], hidden_dims[2])

        # Декодер
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[2] + 1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, bathymetry):
        """
        Args:
            x: (batch_size, seq_len, height, width, channels)
            bathymetry: (batch_size, height, width)
        Returns:
            output: (batch_size, height, width)
        """
        batch_size, seq_len, height, width, channels = x.shape

        # Проверка размерностей
        assert channels == self.input_channels, f"Expected {self.input_channels} channels, got {channels}"
        assert height == self.grid_size[0], f"Expected height {self.grid_size[0]}, got {height}"
        assert width == self.grid_size[1], f"Expected width {self.grid_size[1]}, got {width}"

        # Преобразуем в (batch, seq_len, channels, height, width)
        x = x.permute(0, 1, 4, 2, 3)  # [batch, seq_len, channels, height, width]

        # Инициализация состояний
        device = x.device
        h1 = torch.zeros(batch_size, self.hidden_dims[0], height, width, device=device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(batch_size, self.hidden_dims[1], height, width, device=device)
        c2 = torch.zeros_like(h2)
        h3 = torch.zeros(batch_size, self.hidden_dims[2], height, width, device=device)
        c3 = torch.zeros_like(h3)

        # Проход через время
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]  # [batch, channels, height, width]

            h1, c1 = self.convlstm1(input_t, (h1, c1))
            h2, c2 = self.convlstm2(h1, (h2, c2))
            h3, c3 = self.convlstm3(h2, (h3, c3))

        # Добавляем батиметрию
        # bathymetry: [batch, height, width] -> [batch, 1, height, width]
        if bathymetry.dim() == 3:
            bathymetry = bathymetry.unsqueeze(1)  # [batch, 1, height, width]

        combined = torch.cat([h3, bathymetry], dim=1)  # [batch, hidden_dims[2]+1, height, width]

        # Прогноз
        output = self.decoder(combined)  # [batch, 1, height, width]

        return output.squeeze(1)  # [batch, height, width]

    def predict(self, input_sequence, bathymetry, device='cpu'):
        """Удобный метод для предсказания"""
        print(f"\n[DEBUG] predict method called")
        print(f"[DEBUG] input_sequence type: {type(input_sequence)}")
        print(
            f"[DEBUG] input_sequence shape: {input_sequence.shape if hasattr(input_sequence, 'shape') else 'unknown'}")
        print(f"[DEBUG] bathymetry type: {type(bathymetry)}")
        print(f"[DEBUG] bathymetry shape: {bathymetry.shape if hasattr(bathymetry, 'shape') else 'unknown'}")

        self.eval()

        # Преобразуем в тензоры
        if not isinstance(input_sequence, torch.Tensor):
            print(f"[DEBUG] Converting input_sequence to tensor")
            input_sequence = torch.FloatTensor(input_sequence)
        if not isinstance(bathymetry, torch.Tensor):
            print(f"[DEBUG] Converting bathymetry to tensor")
            bathymetry = torch.FloatTensor(bathymetry)

        print(f"[DEBUG] After conversion - input_sequence shape: {input_sequence.shape}")
        print(f"[DEBUG] After conversion - bathymetry shape: {bathymetry.shape}")

        # Добавляем batch dimension если нужно
        if input_sequence.dim() == 4:  # [seq_len, height, width, channels]
            print(f"[DEBUG] Adding batch dimension to input_sequence")
            input_sequence = input_sequence.unsqueeze(0)  # [1, seq_len, height, width, channels]

        if bathymetry.dim() == 2:  # [height, width]
            print(f"[DEBUG] Adding batch dimension to bathymetry")
            bathymetry = bathymetry.unsqueeze(0)  # [1, height, width]

        print(f"[DEBUG] Final input_sequence shape: {input_sequence.shape}")
        print(f"[DEBUG] Final bathymetry shape: {bathymetry.shape}")

        # Перемещаем на устройство
        input_sequence = input_sequence.to(device)
        bathymetry = bathymetry.to(device)

        with torch.no_grad():
            prediction = self.forward(input_sequence, bathymetry)

        print(f"[DEBUG] Prediction shape: {prediction.shape}")

        return prediction.cpu().numpy()

    def save(self, filepath):
        """Сохранить модель"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_channels': self.input_channels,
            'hidden_dims': self.hidden_dims,
            'grid_size': self.grid_size
        }, filepath)

    @classmethod
    def load(cls, filepath, device='cpu'):
        """Загрузить модель"""
        checkpoint = torch.load(filepath, map_location=device)

        model = cls(
            input_channels=checkpoint['input_channels'],
            hidden_dims=checkpoint['hidden_dims'],
            grid_size=checkpoint['grid_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model