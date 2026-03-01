# test_indexing.py

import numpy as np

print("="*60)
print("ТЕСТ ИНДЕКСАЦИИ МАССИВОВ")
print("="*60)

# Тест 1: Создание 3D массива
print("\n1. Создание 3D массива [time, lat, lon]:")
arr_3d = np.random.rand(100, 30, 60)
print(f"  arr_3d shape: {arr_3d.shape}")
print(f"  arr_3d[0] shape: {arr_3d[0].shape}")
print(f"  arr_3d[0,0] shape: {arr_3d[0,0].shape}")
print(f"  arr_3d[0,0,0] value: {arr_3d[0,0,0]:.3f}")

# Тест 2: Создание 2D массива
print("\n2. Создание 2D массива [lat, lon]:")
arr_2d = np.random.rand(30, 60)
print(f"  arr_2d shape: {arr_2d.shape}")
print(f"  arr_2d[0] shape: {arr_2d[0].shape}")
print(f"  arr_2d[0,0] value: {arr_2d[0,0]:.3f}")

# Тест 3: Проверка присваивания
print("\n3. Проверка присваивания:")
X = np.zeros((1, 10, 30, 60, 7))
print(f"  X shape: {X.shape}")

try:
    X[0, 0, :, :, 0] = arr_3d[0]
    print("  ✓ X[0,0,:,:,0] = arr_3d[0] работает")
except Exception as e:
    print(f"  ✗ Ошибка: {e}")

try:
    X[0, 0, :, :, 0] = arr_2d
    print("  ✓ X[0,0,:,:,0] = arr_2d работает")
except Exception as e:
    print(f"  ✗ Ошибка: {e}")

print("\n" + "="*60)