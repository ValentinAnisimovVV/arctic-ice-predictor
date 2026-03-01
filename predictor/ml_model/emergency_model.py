# predictor/ml_model/emergency_model.py

import numpy as np


class EmergencyModel:
    """Аварийная модель - максимально простая"""

    def __init__(self, grid_size=(30, 60)):
        self.grid_size = grid_size
        print(f"🚨 EmergencyModel создана с grid_size={grid_size}")

    def predict(self, X, bathymetry, device='cpu'):
        """
        Аварийная модель - всегда возвращает осмысленные значения
        """
        print(f"\n[EmergencyModel] predict called")
        print(f"[EmergencyModel] X type: {type(X)}")
        print(f"[EmergencyModel] bathymetry type: {type(bathymetry)}")

        # Преобразуем все в numpy массивы если нужно
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X)
            except:
                X = np.zeros((1, 10, 30, 60, 7))

        if not isinstance(bathymetry, np.ndarray):
            try:
                bathymetry = np.array(bathymetry)
            except:
                bathymetry = np.zeros((30, 60))

        print(f"[EmergencyModel] X shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
        print(f"[EmergencyModel] bathymetry shape: {bathymetry.shape if hasattr(bathymetry, 'shape') else 'unknown'}")

        # Создаем предсказание - простое и надежное
        height, width = self.grid_size

        # Создаем градиент концентрации льда (больше на севере)
        prediction = np.zeros((height, width))
        for i in range(height):
            # Чем севернее (меньше индекс), тем больше льда
            lat_factor = 1.0 - (i / height) * 0.5
            prediction[i, :] = lat_factor * 0.8

        # Добавляем небольшой шум
        prediction += 0.05 * np.random.randn(height, width)
        prediction = np.clip(prediction, 0, 1)

        # Добавляем batch dimension
        prediction = prediction.reshape(1, height, width)

        print(f"[EmergencyModel] prediction shape: {prediction.shape}")
        print(f"[EmergencyModel] prediction min: {prediction.min():.3f}, max: {prediction.max():.3f}")

        return prediction

    def to(self, device):
        return self

    def eval(self):
        return self