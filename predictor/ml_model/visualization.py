# predictor/ml_model/visualization.py

import matplotlib

matplotlib.use('Agg')  # Обязательно в самом начале!
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import uuid


def create_visualization_safe(job_id, input_data, output_data, save_dir):
    """
    Безопасная функция визуализации в отдельном процессе
    """
    try:
        # Явно создаем новую фигуру
        fig = plt.figure(figsize=(12, 10))

        # 2x2 сетка графиков
        ax1 = plt.subplot(2, 2, 1)
        im1 = ax1.imshow(input_data, cmap='Blues_r', vmin=0, vmax=1)
        ax1.set_title('Исходная концентрация льда')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(2, 2, 2)
        im2 = ax2.imshow(output_data, cmap='Blues_r', vmin=0, vmax=1)
        ax2.set_title('Прогнозируемая концентрация')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(2, 2, 3)
        diff = output_data - input_data
        im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax3.set_title('Изменение (прогноз - исходное)')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(input_data.flatten(), bins=50, alpha=0.5, label='Исходное', density=True)
        ax4.hist(output_data.flatten(), bins=50, alpha=0.5, label='Прогноз', density=True)
        ax4.set_xlabel('Концентрация льда')
        ax4.set_ylabel('Плотность')
        ax4.set_title('Распределение концентрации')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Результаты прогнозирования (ID: {job_id})', fontsize=16)
        plt.tight_layout()

        # Сохраняем
        plot_path = Path(save_dir) / f'{job_id}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        # Явно закрываем всё
        plt.close(fig)
        plt.close('all')

        return str(plot_path)

    except Exception as e:
        print(f"Error in visualization: {e}")
        plt.close('all')
        return None
    finally:
        # Гарантированно закрываем
        plt.close('all')