# test_app_direct.py

import os
import sys
import django
import numpy as np

print("=" * 80)
print("ТЕСТ ПРИЛОЖЕНИЯ НАПРЯМУЮ")
print("=" * 80)

# Настраиваем Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'arctic_ice.settings')
django.setup()

print("\n1. ИМПОРТ МОДУЛЕЙ")
print("-" * 40)
from predictor.ml_model.data_generator import ArcticDataGenerator
from predictor.models import PredictionJob

print("✓ Модули импортированы")

print("\n2. ТЕСТ ГЕНЕРАТОРА ДАННЫХ")
print("-" * 40)
generator = ArcticDataGenerator(grid_size=(30, 60), time_steps=100)
print(f"✓ Генератор создан")

print("\n3. ТЕСТ create_sample_sequence")
print("-" * 40)
try:
    X, bathy = generator.create_sample_sequence(sequence_length=10)
    print(f"✓ X shape: {X.shape}")
    print(f"✓ bathy shape: {bathy.shape}")

    # Проверяем каждый канал
    print(f"\n  Проверка каналов:")
    for c in range(7):
        channel_data = X[0, 0, :, :, c]
        print(f"    Канал {c}: shape={channel_data.shape}, min={channel_data.min():.3f}, max={channel_data.max():.3f}")

except Exception as e:
    print(f"✗ Ошибка: {e}")
    import traceback

    traceback.print_exc()

print("\n4. ТЕСТ СОЗДАНИЯ ПРЕДСКАЗАНИЯ")
print("-" * 40)
try:
    # Создаем простую модель прямо здесь
    class TestModel:
        def predict(self, X, bathy, device='cpu'):
            print(f"    TestModel.predict вызван")
            print(f"    X shape: {X.shape}")
            print(f"    bathy shape: {bathy.shape}")

            # Пробуем разные операции
            try:
                # Пытаемся получить данные
                last_step = X[0, -1, :, :, 0]
                print(f"    last_step shape: {last_step.shape}")

                # Создаем предсказание
                pred = np.ones((1, 30, 60)) * 0.5
                print(f"    pred shape: {pred.shape}")

                return pred
            except Exception as e:
                print(f"    ✗ Ошибка внутри predict: {e}")
                raise

        def to(self, device):
            return self

        def eval(self):
            return self


    model = TestModel()

    # Пробуем сделать предсказание
    print(f"\n  Вызов model.predict:")
    prediction = model.predict(X, bathy)
    print(f"  prediction shape: {prediction.shape}")
    print("✓ Предсказание успешно")

except Exception as e:
    print(f"✗ Ошибка: {e}")
    traceback.print_exc()

print("\n5. ТЕСТ ЧЕРЕЗ МОДЕЛЬ ИЗ ПРИЛОЖЕНИЯ")
print("-" * 40)
try:
    from predictor.ml_model.model import ArcticIcePredictor

    print("  Загрузка модели...")
    model2 = ArcticIcePredictor(grid_size=(30, 60))
    print(f"  ✓ Модель создана")

    print(f"  Вызов model2.predict...")
    prediction2 = model2.predict(X, bathy)
    print(f"  prediction2 shape: {prediction2.shape}")

except Exception as e:
    print(f"✗ Ошибка: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
print("ТЕСТ ЗАВЕРШЕН")
print("=" * 80)