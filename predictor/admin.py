# predictor/admin.py

from django.contrib import admin
from .models import PredictionJob, TrainedModel, ForecastResult


@admin.register(PredictionJob)
class PredictionJobAdmin(admin.ModelAdmin):
    list_display = ('job_id', 'created_at', 'status', 'sequence_length', 'forecast_horizon', 'mse')
    list_filter = ('status', 'created_at')
    search_fields = ('job_id',)
    readonly_fields = ('job_id', 'created_at', 'updated_at')
    fieldsets = (
        ('Основная информация', {
            'fields': ('job_id', 'user', 'status', 'created_at', 'updated_at')
        }),
        ('Параметры', {
            'fields': ('sequence_length', 'forecast_horizon')
        }),
        ('Результаты', {
            'fields': ('mse', 'mae', 'rmse', 'input_data_path', 'output_data_path', 'plot_path')
        }),
        ('Логи', {
            'fields': ('error_message', 'logs'),
            'classes': ('collapse',)
        })
    )


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at', 'val_loss', 'test_loss', 'is_active')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name',)
    actions = ['activate_model']

    def activate_model(self, request, queryset):
        # Деактивируем все модели
        TrainedModel.objects.update(is_active=False)
        # Активируем выбранные
        queryset.update(is_active=True)
        self.message_user(request, f"{queryset.count()} моделей активировано")

    activate_model.short_description = "Активировать выбранные модели"


@admin.register(ForecastResult)
class ForecastResultAdmin(admin.ModelAdmin):
    list_display = ('job', 'date', 'mean_ice_concentration', 'ice_area', 'ice_edge_latitude')
    list_filter = ('date', 'job__job_id')
    search_fields = ('job__job_id',)