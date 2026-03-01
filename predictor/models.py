# predictor/models.py

from django.db import models
from django.contrib.auth.models import User
import json


class PredictionJob(models.Model):
    """Модель для отслеживания задач прогнозирования"""

    STATUS_CHOICES = [
        ('pending', 'В очереди'),
        ('processing', 'Обрабатывается'),
        ('completed', 'Завершено'),
        ('failed', 'Ошибка'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    job_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Параметры прогноза
    sequence_length = models.IntegerField(default=30, help_text='Длина входной последовательности (дни)')
    forecast_horizon = models.IntegerField(default=30, help_text='Горизонт прогноза (дни)')

    # Результаты
    input_data_path = models.CharField(max_length=500, blank=True, null=True)
    output_data_path = models.CharField(max_length=500, blank=True, null=True)
    plot_path = models.CharField(max_length=500, blank=True, null=True)

    # Метрики
    mse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)

    # Логи
    error_message = models.TextField(blank=True, null=True)
    logs = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction {self.job_id} - {self.status}"


class TrainedModel(models.Model):
    """Модель для сохранения обученных моделей"""

    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    file_path = models.CharField(max_length=500)

    # Параметры модели
    input_channels = models.IntegerField(default=7)
    hidden_dims = models.CharField(max_length=100, default='[16,32,16]',
                                   help_text='JSON строка с размерностями скрытых слоев')

    # Метрики качества
    val_loss = models.FloatField(null=True, blank=True)
    test_loss = models.FloatField(null=True, blank=True)

    # Активна ли модель
    is_active = models.BooleanField(default=False)

    def get_hidden_dims(self):
        """Получить размерности скрытых слоев как список"""
        return json.loads(self.hidden_dims)

    def __str__(self):
        return f"{self.name} ({self.created_at.strftime('%Y-%m-%d')})"


class ForecastResult(models.Model):
    """Результаты прогноза для конкретной даты"""

    job = models.ForeignKey(PredictionJob, on_delete=models.CASCADE, related_name='results')
    date = models.DateField()

    # Средние значения по региону
    mean_ice_concentration = models.FloatField()
    min_ice_concentration = models.FloatField()
    max_ice_concentration = models.FloatField()

    # Площадь льда (в млн км²)
    ice_area = models.FloatField(help_text='Площадь льда в млн км²')

    # Граница льда (средняя широта)
    ice_edge_latitude = models.FloatField(help_text='Средняя широта границы льда')

    # JSON с полными данными для визуализации
    full_data = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['date']
        unique_together = ['job', 'date']

    def __str__(self):
        return f"{self.date} - {self.mean_ice_concentration:.2f}"