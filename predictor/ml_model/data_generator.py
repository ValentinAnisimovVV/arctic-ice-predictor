# predictor/ml_model/data_generator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from tqdm import tqdm


class ArcticDataGenerator:
    """Генератор синтетических данных об арктическом льде"""

    def __init__(self, grid_size=(30, 60), time_steps=365, seed=42):
        """
        Args:
            grid_size: (широты, долготы)
            time_steps: количество временных шагов (дней)
            seed: random seed для воспроизводимости
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.lats = np.linspace(60, 90, grid_size[0])
        self.lons = np.linspace(-180, 180, grid_size[1])
        self.time_index = pd.date_range('2020-01-01', periods=time_steps, freq='D')
        np.random.seed(seed)

    def generate_sea_ice_concentration(self):
        """Генерация концентрации морского льда с сезонными циклами и трендом"""
        print("Генерация концентрации морского льда...")

        # Создаем 3D массив [time, lat, lon]
        ice_conc = np.zeros((self.time_steps, self.grid_size[0], self.grid_size[1]), dtype=np.float32)

        t_array = np.arange(self.time_steps)
        season_factor = 0.7 + 0.3 * np.cos(2 * np.pi * t_array / 365)

        # Правильно расширяем lat_factor для умножения
        lat_factor = np.zeros((self.grid_size[0], self.grid_size[1]))
        for i in range(self.grid_size[0]):
            lat_factor[i, :] = 1 - (self.lats[i] - 60) / 30

        trend = 0.95 - 0.0005 * t_array

        for t in range(self.time_steps):
            base = 0.8 * season_factor[t] * lat_factor
            spatial_noise = 0.1 * np.random.randn(self.grid_size[0], self.grid_size[1]).astype(np.float32)
            ice_conc[t] = np.clip(base * trend[t] + spatial_noise, 0, 1)

        print(f"  ✓ ice_conc shape: {ice_conc.shape}")
        return ice_conc

    def generate_temperature(self):
        """Генерация температуры поверхности с широтным градиентом"""
        print("Генерация температуры поверхности...")

        temperature = np.zeros((self.time_steps, self.grid_size[0], self.grid_size[1]), dtype=np.float32)

        t_array = np.arange(self.time_steps)
        season = -25 + 30 * np.sin(2 * np.pi * t_array / 365 - np.pi / 2)

        # Создаем широтный градиент
        lat_grad = np.zeros((self.grid_size[0], self.grid_size[1]))
        for i in range(self.grid_size[0]):
            lat_grad[i, :] = -0.8 * (self.lats[i] - 75)

        for t in range(self.time_steps):
            temperature[t] = season[t] + lat_grad + 5 * np.random.randn(self.grid_size[0], self.grid_size[1]).astype(
                np.float32)

        print(f"  ✓ temperature shape: {temperature.shape}")
        return temperature

    def generate_bathymetry(self):
        """Генерация реалистичной батиметрии Арктики"""
        print("Генерация батиметрии...")

        bathymetry = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.float32)

        # Создаем сетку координат
        x, y = np.meshgrid(self.lons, self.lats)

        # Основной рельеф
        bathymetry = -4000 + 3000 * np.sin(0.02 * x) * np.cos(0.02 * y)

        # Хребет Ломоносова
        ridge_mask = (np.abs(y - 85) < 3) & (np.abs(x) < 40)
        bathymetry[ridge_mask] += 2000

        # Шельфы
        shelf_mask = (y < 70) & (np.abs(x) > 120)
        bathymetry[shelf_mask] = -150

        # Глубоководные котловины
        basin_mask = (y > 80) & (np.abs(x) > 60)
        bathymetry[basin_mask] -= 1000

        print(f"  ✓ bathymetry shape: {bathymetry.shape}")
        return bathymetry

    def generate_wind_fields(self):
        """Генерация полей ветра с арктической циркуляцией"""
        print("Генерация полей ветра...")

        u_wind = np.zeros((self.time_steps, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        v_wind = np.zeros((self.time_steps, self.grid_size[0], self.grid_size[1]), dtype=np.float32)

        x_grid, y_grid = np.meshgrid(self.lons, self.lats)

        for t in range(self.time_steps):
            # Основная циркуляция
            u_wind[t] = -12 * np.sin(0.08 * x_grid) * np.cos(0.08 * y_grid)
            v_wind[t] = 10 * np.cos(0.08 * x_grid) * np.sin(0.08 * y_grid)

            # Сезонные вариации
            season_factor = 0.8 + 0.2 * np.sin(2 * np.pi * t / 365)
            u_wind[t] *= season_factor
            v_wind[t] *= season_factor

            # Турбулентность
            u_wind[t] += 4 * np.random.randn(self.grid_size[0], self.grid_size[1]).astype(np.float32)
            v_wind[t] += 4 * np.random.randn(self.grid_size[0], self.grid_size[1]).astype(np.float32)

        print(f"  ✓ u_wind shape: {u_wind.shape}")
        print(f"  ✓ v_wind shape: {v_wind.shape}")
        return u_wind, v_wind

    def generate_ocean_currents(self):
        """Генерация океанических течений"""
        print("Генерация океанических течений...")

        u_current = np.zeros((self.time_steps, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        v_current = np.zeros((self.time_steps, self.grid_size[0], self.grid_size[1]), dtype=np.float32)

        x_grid, y_grid = np.meshgrid(self.lons, self.lats)

        for t in range(self.time_steps):
            # Трансполярное течение
            u_current[t] = 0.15 * np.sin(0.03 * x_grid) * np.cos(0.05 * y_grid)
            v_current[t] = 0.08 * np.cos(0.03 * x_grid) * np.sin(0.05 * y_grid)

            # Случайные возмущения
            u_current[t] += 0.03 * np.random.randn(self.grid_size[0], self.grid_size[1]).astype(np.float32)
            v_current[t] += 0.03 * np.random.randn(self.grid_size[0], self.grid_size[1]).astype(np.float32)

        print(f"  ✓ u_current shape: {u_current.shape}")
        print(f"  ✓ v_current shape: {v_current.shape}")
        return u_current, v_current

    def generate_full_dataset(self):
        """Генерация полного набора данных"""
        print("\n" + "=" * 60)
        print("ГЕНЕРАЦИЯ ПОЛНОГО НАБОРА ДАННЫХ АРКТИКИ")
        print("=" * 60)

        dataset = {
            'sea_ice_concentration': self.generate_sea_ice_concentration(),
            'surface_temperature': self.generate_temperature(),
            'bathymetry': self.generate_bathymetry(),
        }

        # Добавляем ветер и течения
        u_wind, v_wind = self.generate_wind_fields()
        u_current, v_current = self.generate_ocean_currents()

        dataset['u_wind'] = u_wind
        dataset['v_wind'] = v_wind
        dataset['u_current'] = u_current
        dataset['v_current'] = v_current

        # Целевая переменная: концентрация льда через 30 дней
        future_ice = np.roll(dataset['sea_ice_concentration'], -30, axis=0)
        future_ice[-30:] = dataset['sea_ice_concentration'][-1]  # Заполняем последние 30 дней последним значением
        dataset['future_sea_ice'] = future_ice

        return dataset

    def create_sample_sequence(self, sequence_length=30):
        """Создать пример последовательности для демонстрации"""
        print(f"\n[DEBUG] create_sample_sequence started with sequence_length={sequence_length}")

        dataset = self.generate_full_dataset()

        # Проверяем размерности данных
        print(f"[DEBUG] dataset keys: {list(dataset.keys())}")
        print(f"[DEBUG] sea_ice_concentration shape: {dataset['sea_ice_concentration'].shape}")
        print(f"[DEBUG] surface_temperature shape: {dataset['surface_temperature'].shape}")
        print(f"[DEBUG] bathymetry shape: {dataset['bathymetry'].shape}")

        # Берем случайный временной срез
        max_start = self.time_steps - sequence_length - 30
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)

        print(f"[DEBUG] start_idx={start_idx}")

        # Создаем входные данные: [batch, seq_len, height, width, channels]
        height, width = self.grid_size

        # Инициализируем массив правильной формы
        X = np.zeros((1, sequence_length, height, width, 7), dtype=np.float32)

        # Заполняем данные
        for t in range(sequence_length):
            idx = start_idx + t

            # Проверяем, что idx в допустимых пределах
            if idx >= self.time_steps:
                idx = self.time_steps - 1

            # Концентрация льда - должно быть 2D
            ice_data = dataset['sea_ice_concentration'][idx]
            if ice_data.shape != (height, width):
                print(f"  ⚠ ice_data has wrong shape: {ice_data.shape}, expected ({height}, {width})")
                # Пытаемся исправить
                if len(ice_data.shape) == 1:
                    # Если одномерный, превращаем в 2D
                    ice_data = ice_data.reshape(height, width)
            X[0, t, :, :, 0] = ice_data

            # Температура - должно быть 2D
            temp_data = dataset['surface_temperature'][idx]
            if temp_data.shape != (height, width):
                print(f"  ⚠ temp_data has wrong shape: {temp_data.shape}, expected ({height}, {width})")
                if len(temp_data.shape) == 1:
                    temp_data = temp_data.reshape(height, width)
            X[0, t, :, :, 1] = temp_data

            # Ветер u (если есть)
            if 'u_wind' in dataset:
                wind_data = dataset['u_wind'][idx]
                if wind_data.shape != (height, width):
                    if len(wind_data.shape) == 1:
                        wind_data = wind_data.reshape(height, width)
                X[0, t, :, :, 2] = wind_data
            else:
                X[0, t, :, :, 2] = np.zeros((height, width))

            # Ветер v
            if 'v_wind' in dataset:
                wind_data = dataset['v_wind'][idx]
                if wind_data.shape != (height, width):
                    if len(wind_data.shape) == 1:
                        wind_data = wind_data.reshape(height, width)
                X[0, t, :, :, 3] = wind_data
            else:
                X[0, t, :, :, 3] = np.zeros((height, width))

            # Течение u
            if 'u_current' in dataset:
                current_data = dataset['u_current'][idx]
                if current_data.shape != (height, width):
                    if len(current_data.shape) == 1:
                        current_data = current_data.reshape(height, width)
                X[0, t, :, :, 4] = current_data
            else:
                X[0, t, :, :, 4] = np.zeros((height, width))

            # Течение v
            if 'v_current' in dataset:
                current_data = dataset['v_current'][idx]
                if current_data.shape != (height, width):
                    if len(current_data.shape) == 1:
                        current_data = current_data.reshape(height, width)
                X[0, t, :, :, 5] = current_data
            else:
                X[0, t, :, :, 5] = np.zeros((height, width))

            # Батиметрия (одинаковая для всех t)
            bathy_data = dataset['bathymetry']
            if bathy_data.shape != (height, width):
                print(f"  ⚠ bathy_data has wrong shape: {bathy_data.shape}, expected ({height}, {width})")
                if len(bathy_data.shape) == 1:
                    bathy_data = bathy_data.reshape(height, width)
            X[0, t, :, :, 6] = bathy_data

        # Проверяем финальные размерности
        print(f"[DEBUG] Final X shape: {X.shape}")
        print(f"[DEBUG] Final X min: {X.min():.3f}, max: {X.max():.3f}")
        print(f"[DEBUG] Final bathymetry shape: {dataset['bathymetry'].shape}")

        return X, dataset['bathymetry']

    def _calculate_ice_edge(self, ice_map):
        """Вычислить среднюю широту границы льда"""
        ice_threshold = 0.15
        ice_present = ice_map > ice_threshold

        ice_edge_lats = []
        for lon_idx in range(ice_present.shape[1]):
            ice_in_column = np.where(ice_present[:, lon_idx])[0]
            if len(ice_in_column) > 0:
                ice_edge_lats.append(self.lats[ice_in_column[0]])

        if ice_edge_lats:
            return float(np.mean(ice_edge_lats))
        else:
            return 90.0