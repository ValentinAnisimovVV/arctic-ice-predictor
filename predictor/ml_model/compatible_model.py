# predictor/ml_model/compatible_model.py

"""
Совместимая модель ConvLSTM для прогнозирования арктического льда
Работает с NumPy 1.24.3 и PyTorch 2.1.0
"""

import torch
import torch.nn as nn
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class ConvLSTMCell(nn.Module):
    """
    Ячейка ConvLSTM для пространственно-временного моделирования
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        """
        Args:
            input_dim: число каналов на входе
            hidden_dim: число каналов в скрытом состоянии
            kernel_size: размер ядра свертки
            bias: использовать ли смещение
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Свертка для объединенного входа (x + h)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, cur_state):
        """
        Прямой проход через ячейку

        Args:
            x: входной тензор [batch, input_dim, height, width]
            cur_state: текущее состояние (h, c)

        Returns:
            h_next: следующее скрытое состояние
            c_next: следующее состояние памяти
        """
        h_cur, c_cur = cur_state

        # Объединяем вход и скрытое состояние
        combined = torch.cat([x, h_cur], dim=1)

        # Применяем свертку
        gates = self.conv(combined)

        # Разделяем на вентили
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        # Применяем активации
        i = torch.sigmoid(i)  # входной вентиль
        f = torch.sigmoid(f)  # вентиль забывания
        o = torch.sigmoid(o)  # выходной вентиль
        g = torch.tanh(g)  # кандидат на обновление

        # Обновляем состояние памяти
        c_next = f * c_cur + i * g

        # Обновляем скрытое состояние
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Инициализация скрытого состояния

        Args:
            batch_size: размер батча
            image_size: размер изображения (height, width)

        Returns:
            tuple: (h, c) - начальные состояния
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class CompatibleArcticModel(nn.Module):
    """
    Совместимая модель ConvLSTM для прогнозирования арктического льда

    Архитектура:
    - 3 слоя ConvLSTM для извлечения пространственно-временных признаков
    - Декодер для преобразования признаков в карту концентрации льда
    """

    def __init__(self, input_channels=7, hidden_dims=[16, 32, 16],
                 kernel_size=3, grid_size=(30, 60)):
        """
        Args:
            input_channels: количество входных каналов (ice, temp, wind, currents, bathymetry)
            hidden_dims: размерности скрытых слоев
            kernel_size: размер ядра свертки
            grid_size: размер сетки (широта, долгота)
        """
        super(CompatibleArcticModel, self).__init__()

        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.num_layers = len(hidden_dims)

        # Список слоев ConvLSTM
        cell_list = []
        for i in range(self.num_layers):
            in_channels = self.input_channels if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channels, self.hidden_dims[i], self.kernel_size)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Декодер для получения финального предсказания
        # Вход: последнее скрытое состояние + батиметрия
        decoder_input_channels = self.hidden_dims[-1] + 1  # +1 для батиметрии

        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=1),  # 1x1 свертка для получения одного канала
            nn.Sigmoid()  # Ограничиваем значения [0, 1]
        )

        print(f"✅ CompatibleArcticModel создана")
        print(f"   Входные каналы: {input_channels}")
        print(f"   Скрытые слои: {hidden_dims}")
        print(f"   Размер сетки: {grid_size}")
        print(f"   Параметров: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x, bathymetry):
        """
        Прямой проход через модель

        Args:
            x: входная последовательность [batch, seq_len, height, width, channels]
            bathymetry: карта батиметрии [batch, height, width]

        Returns:
            output: предсказанная концентрация льда [batch, height, width]
        """
        batch_size, seq_len, height, width, channels = x.shape

        # Проверка размерностей
        assert channels == self.input_channels, \
            f"Ожидалось {self.input_channels} каналов, получено {channels}"
        assert height == self.grid_size[0] and width == self.grid_size[1], \
            f"Ожидался размер {self.grid_size}, получен ({height}, {width})"

        # Переставляем размерности: [batch, seq_len, height, width, channels] ->
        # [batch, seq_len, channels, height, width]
        x = x.permute(0, 1, 4, 2, 3)

        # Инициализируем состояния для всех слоев
        h_states = []
        c_states = []

        for i in range(self.num_layers):
            h, c = self.cell_list[i].init_hidden(batch_size, (height, width))
            h_states.append(h)
            c_states.append(c)

        # Проходим по временным шагам
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]  # [batch, channels, height, width]

            for i in range(self.num_layers):
                if i == 0:
                    h_states[i], c_states[i] = self.cell_list[i](x_t, (h_states[i], c_states[i]))
                else:
                    h_states[i], c_states[i] = self.cell_list[i](h_states[i - 1], (h_states[i], c_states[i]))

        # Берем последнее скрытое состояние верхнего слоя
        last_hidden = h_states[-1]  # [batch, hidden_dim, height, width]

        # Добавляем батиметрию
        # bathymetry: [batch, height, width] -> [batch, 1, height, width]
        if bathymetry.dim() == 3:
            bathymetry = bathymetry.unsqueeze(1)

        # Объединяем признаки
        combined = torch.cat([last_hidden, bathymetry], dim=1)  # [batch, hidden_dim+1, height, width]

        # Применяем декодер
        output = self.decoder(combined)  # [batch, 1, height, width]

        # Убираем лишнюю размерность канала
        output = output.squeeze(1)  # [batch, height, width]

        return output

    def predict(self, X, bathymetry, device='cpu'):
        """
        Удобный метод для предсказания

        Args:
            X: входные данные (numpy array или torch tensor)
               форма: [batch, seq_len, height, width, channels] или [seq_len, height, width, channels]
            bathymetry: карта батиметрии (numpy array или torch tensor)
                       форма: [batch, height, width] или [height, width]
            device: устройство для вычислений ('cpu' или 'cuda')

        Returns:
            prediction: numpy array с предсказанием [batch, height, width]
        """
        print(f"\n[CompatibleModel] predict вызван")

        # Переводим модель в режим оценки
        self.eval()

        # Перемещаем модель на нужное устройство
        self.to(device)

        # Конвертация numpy -> torch с проверкой
        if not isinstance(X, torch.Tensor):
            try:
                X = torch.tensor(X, dtype=torch.float32)
                print(f"  ✓ X сконвертирован в tensor, форма: {X.shape}")
            except Exception as e:
                print(f"  ✗ Ошибка конвертации X: {e}")
                raise

        if not isinstance(bathymetry, torch.Tensor):
            try:
                bathymetry = torch.tensor(bathymetry, dtype=torch.float32)
                print(f"  ✓ bathymetry сконвертирован в tensor, форма: {bathymetry.shape}")
            except Exception as e:
                print(f"  ✗ Ошибка конвертации bathymetry: {e}")
                raise

        # Добавляем batch dimension если нужно
        if X.dim() == 4:  # [seq_len, height, width, channels]
            X = X.unsqueeze(0)
            print(f"  ✓ Добавлен batch dimension к X: {X.shape}")

        if bathymetry.dim() == 2:  # [height, width]
            bathymetry = bathymetry.unsqueeze(0)
            print(f"  ✓ Добавлен batch dimension к bathymetry: {bathymetry.shape}")

        # Проверка размерностей
        expected_seq_len = X.shape[1]
        expected_height, expected_width = self.grid_size

        assert X.shape[2] == expected_height and X.shape[3] == expected_width, \
            f"Неверный размер сетки: ожидалось ({expected_height}, {expected_width}), получено ({X.shape[2]}, {X.shape[3]})"

        assert X.shape[4] == self.input_channels, \
            f"Неверное число каналов: ожидалось {self.input_channels}, получено {X.shape[4]}"

        # Перемещаем на устройство
        X = X.to(device)
        bathymetry = bathymetry.to(device)

        print(f"  ✓ Данные перемещены на {device}")
        print(f"  ✓ Финальная форма X: {X.shape}")
        print(f"  ✓ Финальная форма bathymetry: {bathymetry.shape}")

        # Предсказание без градиентов
        with torch.no_grad():
            prediction = self.forward(X, bathymetry)

        print(f"  ✓ Предсказание получено, форма: {prediction.shape}")

        # Конвертация torch -> numpy
        try:
            result = prediction.cpu().detach().numpy()
            print(f"  ✓ Результат сконвертирован в numpy, форма: {result.shape}")
            return result
        except Exception as e:
            print(f"  ✗ Ошибка конвертации в numpy: {e}")
            # Альтернативный метод конвертации
            result = prediction.cpu().numpy()
            return result

    def save(self, filepath):
        """
        Сохранить модель

        Args:
            filepath: путь для сохранения
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_channels': self.input_channels,
            'hidden_dims': self.hidden_dims,
            'kernel_size': self.kernel_size,
            'grid_size': self.grid_size
        }, filepath)
        print(f"✅ Модель сохранена в {filepath}")

    @classmethod
    def load(cls, filepath, device='cpu'):
        """
        Загрузить модель

        Args:
            filepath: путь к сохраненной модели
            device: устройство для загрузки

        Returns:
            model: загруженная модель
        """
        checkpoint = torch.load(filepath, map_location=device)

        model = cls(
            input_channels=checkpoint['input_channels'],
            hidden_dims=checkpoint['hidden_dims'],
            kernel_size=checkpoint.get('kernel_size', 3),
            grid_size=checkpoint['grid_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"✅ Модель загружена из {filepath}")
        return model


# Простая тестовая функция
def test_model():
    """Тестирование модели"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 60)

    # Параметры
    batch_size = 2
    seq_len = 10
    height, width = 30, 60
    channels = 7

    # Создаем модель
    model = CompatibleArcticModel(
        input_channels=channels,
        hidden_dims=[8, 16, 8],  # Уменьшенные размеры для теста
        grid_size=(height, width)
    )

    print(f"\n1. Модель создана")

    # Создаем тестовые данные
    X = torch.randn(batch_size, seq_len, height, width, channels)
    bathymetry = torch.randn(batch_size, height, width)

    print(f"2. Тестовые данные:")
    print(f"   X shape: {X.shape}")
    print(f"   bathymetry shape: {bathymetry.shape}")

    # Прямой проход
    try:
        output = model(X, bathymetry)
        print(f"3. Прямой проход успешен")
        print(f"   output shape: {output.shape}")
        print(f"   output min: {output.min().item():.3f}")
        print(f"   output max: {output.max().item():.3f}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return

    # Тест predict метода
    try:
        # Конвертируем в numpy
        X_np = X.numpy()
        bathy_np = bathymetry.numpy()

        prediction = model.predict(X_np, bathy_np)
        print(f"4. Predict метод успешен")
        print(f"   prediction shape: {prediction.shape}")
        print(f"   prediction type: {type(prediction)}")
    except Exception as e:
        print(f"   ❌ Ошибка в predict: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 60)

    return model


if __name__ == "__main__":
    # Запуск теста при прямом вызове
    test_model()