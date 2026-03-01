# debug_step_by_step.py

import sys
import os
import numpy as np
import traceback

print("=" * 80)
print("ПОШАГОВАЯ ОТЛАДКА ПРОБЛЕМЫ С РАЗМЕРНОСТЯМИ")
print("=" * 80)

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def debug_data_generator_step_by_step():
    """Пошаговая отладка генератора данных"""

    print("\n1. ИМПОРТ МОДУЛЯ")
    print("-" * 40)
    try:
        from predictor.ml_model.data_generator import ArcticDataGenerator
        print("✓ Модуль импортирован успешно")
    except Exception as e:
        print(f"✗ Ошибка импорта: {e}")
        traceback.print_exc()
        return

    print("\n2. СОЗДАНИЕ ГЕНЕРАТОРА")
    print("-" * 40)
    try:
        generator = ArcticDataGenerator(grid_size=(5, 10), time_steps=20)
        print(f"✓ Генератор создан")
        print(f"  grid_size: {generator.grid_size}")
        print(f"  time_steps: {generator.time_steps}")
        print(f"  lats shape: {generator.lats.shape}")
        print(f"  lons shape: {generator.lons.shape}")
    except Exception as e:
        print(f"✗ Ошибка создания генератора: {e}")
        traceback.print_exc()
        return

    print("\n3. ТЕСТ generate_sea_ice_concentration()")
    print("-" * 40)
    try:
        ice_conc = generator.generate_sea_ice_concentration()
        print(f"✓ ice_conc shape: {ice_conc.shape}")
        print(f"  Тип данных: {ice_conc.dtype}")
        print(f"  Минимум: {ice_conc.min():.3f}")
        print(f"  Максимум: {ice_conc.max():.3f}")

        # Проверяем индексацию
        print("\n  Проверка индексации ice_conc[0]:")
        try:
            val = ice_conc[0]
            print(f"    ice_conc[0] shape: {val.shape}")
            print(f"    Тип: {type(val)}")
        except Exception as e:
            print(f"    ✗ Ошибка: {e}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        traceback.print_exc()

    print("\n4. ТЕСТ generate_temperature()")
    print("-" * 40)
    try:
        temp = generator.generate_temperature()
        print(f"✓ temp shape: {temp.shape}")

        # Проверяем индексацию
        try:
            val = temp[0]
            print(f"  temp[0] shape: {val.shape}")
        except Exception as e:
            print(f"  ✗ Ошибка индексации temp[0]: {e}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")

    print("\n5. ТЕСТ generate_bathymetry()")
    print("-" * 40)
    try:
        bathy = generator.generate_bathymetry()
        print(f"✓ bathy shape: {bathy.shape}")
        print(f"  bathy тип: {type(bathy)}")
        print(f"  bathy размерность: {bathy.ndim}")

        # Проверяем индексацию
        try:
            if bathy.ndim == 2:
                val = bathy[0, 0]
                print(f"  bathy[0,0] = {val:.3f}")
            else:
                print(f"  ⚠ bathy имеет {bathy.ndim} измерений, ожидалось 2")
        except Exception as e:
            print(f"  ✗ Ошибка индексации bathy: {e}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")

    print("\n6. ТЕСТ generate_full_dataset()")
    print("-" * 40)
    try:
        dataset = generator.generate_full_dataset()
        print("✓ Датасет создан")

        print("\n  Проверка каждого элемента датасета:")
        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, ndim={value.ndim}, dtype={value.dtype}")

                # Проверяем возможность индексации
                if value.ndim >= 1:
                    try:
                        if value.ndim == 1:
                            val = value[0]
                            print(f"    → value[0] = {val:.3f} (1D)")
                        elif value.ndim == 2:
                            val = value[0, 0]
                            print(f"    → value[0,0] = {val:.3f} (2D)")
                        elif value.ndim == 3:
                            val = value[0, 0, 0]
                            print(f"    → value[0,0,0] = {val:.3f} (3D)")
                    except Exception as e:
                        print(f"    ✗ Ошибка индексации: {e}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        traceback.print_exc()

    print("\n7. ТЕСТ create_sample_sequence() - ПОШАГОВЫЙ")
    print("-" * 40)

    sequence_length = 5

    # Сначала создаем датасет отдельно
    dataset = generator.generate_full_dataset()

    print(f"\n  Параметры:")
    print(f"  sequence_length = {sequence_length}")
    print(f"  grid_size = {generator.grid_size}")
    print(f"  time_steps = {generator.time_steps}")

    # Выбираем start_idx
    max_start = generator.time_steps - sequence_length - 30
    start_idx = 0 if max_start <= 0 else np.random.randint(0, max_start)
    print(f"  start_idx = {start_idx}")

    height, width = generator.grid_size

    # Создаем массив X
    print(f"\n  Создание массива X с формой (1, {sequence_length}, {height}, {width}, 7)")
    X = np.zeros((1, sequence_length, height, width, 7), dtype=np.float32)
    print(f"  ✓ X создан, shape={X.shape}")

    # Заполняем данные
    print(f"\n  Заполнение данных:")
    for t in range(sequence_length):
        idx = start_idx + t
        if idx >= generator.time_steps:
            idx = generator.time_steps - 1

        print(f"\n  --- t={t}, idx={idx} ---")

        # 1. Концентрация льда
        try:
            ice_data = dataset['sea_ice_concentration'][idx]
            print(
                f"    ice_data: shape={ice_data.shape if hasattr(ice_data, 'shape') else 'scalar'}, type={type(ice_data)}")

            if isinstance(ice_data, np.ndarray):
                if ice_data.shape == (height, width):
                    X[0, t, :, :, 0] = ice_data
                    print(f"    ✓ ice_data присвоено")
                elif ice_data.ndim == 1:
                    print(f"    ⚠ ice_data 1D, reshaping...")
                    ice_data = ice_data.reshape(height, width)
                    X[0, t, :, :, 0] = ice_data
                    print(f"    ✓ ice_data reshaped и присвоено")
                else:
                    print(f"    ⚠ ice_data имеет форму {ice_data.shape}, ожидалось ({height}, {width})")
            else:
                print(f"    ⚠ ice_data не является массивом numpy")
        except Exception as e:
            print(f"    ✗ Ошибка с ice_data: {e}")
            traceback.print_exc()

        # 2. Температура
        try:
            temp_data = dataset['surface_temperature'][idx]
            print(f"    temp_data: shape={temp_data.shape if hasattr(temp_data, 'shape') else 'scalar'}")

            if isinstance(temp_data, np.ndarray):
                if temp_data.shape == (height, width):
                    X[0, t, :, :, 1] = temp_data
                    print(f"    ✓ temp_data присвоено")
                elif temp_data.ndim == 1:
                    print(f"    ⚠ temp_data 1D, reshaping...")
                    temp_data = temp_data.reshape(height, width)
                    X[0, t, :, :, 1] = temp_data
                    print(f"    ✓ temp_data reshaped и присвоено")
        except Exception as e:
            print(f"    ✗ Ошибка с temp_data: {e}")

        # 3. Батиметрия (одинаковая для всех t)
        try:
            bathy_data = dataset['bathymetry']
            print(f"    bathy_data: shape={bathy_data.shape if hasattr(bathy_data, 'shape') else 'scalar'}")

            if isinstance(bathy_data, np.ndarray):
                if bathy_data.shape == (height, width):
                    X[0, t, :, :, 6] = bathy_data
                    print(f"    ✓ bathy_data присвоено")
                elif bathy_data.ndim == 1:
                    print(f"    ⚠ bathy_data 1D, reshaping...")
                    bathy_data = bathy_data.reshape(height, width)
                    X[0, t, :, :, 6] = bathy_data
                    print(f"    ✓ bathy_data reshaped и присвоено")
                else:
                    print(f"    ⚠ bathy_data имеет форму {bathy_data.shape}")
        except Exception as e:
            print(f"    ✗ Ошибка с bathy_data: {e}")

    print(f"\n  Финальная проверка X:")
    print(f"  X shape: {X.shape}")
    print(f"  X min: {X.min():.3f}, max: {X.max():.3f}")

    return generator, dataset


if __name__ == "__main__":
    debug_data_generator_step_by_step()

    print("\n" + "=" * 80)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)