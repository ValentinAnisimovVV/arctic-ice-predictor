# predictor/views.py
from django.contrib.admin.views.decorators import staff_member_required
# predictor/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.contrib import messages
from django.conf import settings
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST

from .models import PredictionJob, TrainedModel, ForecastResult
from .ml_model.model import ArcticIcePredictor
from .ml_model.data_generator import ArcticDataGenerator
import numpy as np
import uuid
import json
from datetime import datetime, timedelta
import os
import matplotlib
matplotlib.use('Agg')  # Отключаем Tkinter, используем только для сохранения файлов
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from pathlib import Path
import torch  # <-- ВАЖНО: добавляем этот импорт!
from django.utils import timezone  # <-- ЭТОТ ИМПОРТ НУЖЕН!
from datetime import timedelta
import logging

# Глобальные переменные для модели (загружаем один раз)
_model = None
_generator = None


# predictor/views.py - временно используем аварийную модель

def get_model():
    """Загрузить или создать модель"""
    global _model
    if _model is None:
        try:
            # Сначала пробуем загрузить совместимую модель
            from .ml_model.compatible_model import CompatibleArcticModel
            _model = CompatibleArcticModel(grid_size=(30, 60))
            print("✅ Используется совместимая модель PyTorch")
        except ImportError as e:
            print(f"⚠️ Не удалось загрузить PyTorch модель: {e}")
            # Если не получается, используем простую numpy модель
            from .ml_model.emergency_model import EmergencyModel
            _model = EmergencyModel(grid_size=(30, 60))
            print("✅ Используется простая numpy модель")
        except Exception as e:
            print(f"❌ Ошибка создания модели: {e}")
            # Аварийная модель на самый крайний случай
            class EmergencyModel:
                def predict(self, X, bathy, device='cpu'):
                    print("🚨 Используется аварийная модель")
                    return np.ones((1, 30, 60)) * 0.5
                def to(self, device): return self
                def eval(self): return self
            _model = EmergencyModel()
    return _model

def get_generator():
    """Создать или получить генератор данных"""
    global _generator
    if _generator is None:
        _generator = ArcticDataGenerator(grid_size=(30, 60))
    return _generator

# ... остальной код (index, predict, и т.д.) ...

def index(request):
    """Главная страница"""
    context = {
        'title': 'Главная - Прогнозирование арктического льда',
        'active_page': 'home'
    }
    return render(request, 'predictor/index.html', context)


def predict(request):
    """Страница прогнозирования"""
    if request.method == 'POST':
        try:
            # Получаем параметры из формы
            sequence_length = int(request.POST.get('sequence_length', 30))
            forecast_horizon = int(request.POST.get('forecast_horizon', 30))
            use_pretrained = request.POST.get('use_pretrained') == 'on'

            # Создаем задачу
            job_id = str(uuid.uuid4())[:8]
            job = PredictionJob.objects.create(
                job_id=job_id,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon,
                status='processing'
            )

            # Запускаем прогноз (в реальном приложении лучше через Celery)
            result = run_prediction(job_id, sequence_length, forecast_horizon)

            if result['success']:
                return redirect('predictor:results', job_id=job_id)
            else:
                messages.error(request, f"Ошибка при прогнозировании: {result['error']}")
                return redirect('predictor:predict')

        except Exception as e:
            messages.error(request, f"Ошибка: {str(e)}")
            return redirect('predictor:predict')

    context = {
        'title': 'Прогнозирование',
        'active_page': 'predict'
    }
    return render(request, 'predictor/predict.html', context)


# predictor/views.py (часть с run_prediction)

def run_prediction(job_id, sequence_length, forecast_horizon):

    """Запустить прогнозирование"""
    try:
        # Получаем модель и генератор
        model = get_model()
        generator = get_generator()

        # Получаем задачу
        job = PredictionJob.objects.get(job_id=job_id)

        # Генерируем входные данные с правильными размерностями
        X, bathymetry = generator.create_sample_sequence(sequence_length)

        print(f"DEBUG - X shape: {X.shape}")
        print(f"DEBUG - Bathymetry shape: {bathymetry.shape}")

        # Сохраняем входные данные
        input_path = settings.MEDIA_ROOT / f'predictions/{job_id}_input.npy'
        os.makedirs(input_path.parent, exist_ok=True)
        np.save(input_path, X)
        job.input_data_path = str(input_path)

        # Определяем устройство
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"DEBUG - Using device: {device}")

        # Перемещаем модель на устройство
        model = model.to(device)

        # Прогнозируем
        predictions = model.predict(X, bathymetry, device=device)

        print(f"DEBUG - Predictions shape: {predictions.shape}")

        # Сохраняем результаты
        output_path = settings.MEDIA_ROOT / f'predictions/{job_id}_output.npy'
        np.save(output_path, predictions)
        job.output_data_path = str(output_path)

        # Создаем визуализацию
        plot_path = create_visualization(job_id, X[0, -1, :, :, 0], predictions[0])
        job.plot_path = plot_path

        # Вычисляем метрики
        y_true = X[0, -1, :, :, 0]  # последний известный день
        y_pred = predictions[0]

        mse = float(np.mean((y_pred - y_true) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(mse))

        job.mse = mse
        job.mae = mae
        job.rmse = rmse

        # Создаем записи с прогнозами по дням
        future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, min(forecast_horizon, 10) + 1)]

        for i, date in enumerate(future_dates):
            # Для простоты используем одно и то же предсказание для всех дней
            # В реальности здесь должна быть модель, предсказывающая на каждый день

            # Правильный расчет границы льда
            ice_mask = predictions[0] > 0.15  # 2D булева маска

            if np.any(ice_mask):
                # Для каждой долготы находим самую южную широту со льдом
                ice_edge_lats = []
                for lon_idx in range(ice_mask.shape[1]):
                    lat_indices = np.where(ice_mask[:, lon_idx])[0]
                    if len(lat_indices) > 0:
                        # Самая южная широта (максимальный индекс)
                        southernmost_lat_idx = lat_indices[-1]
                        ice_edge_lats.append(generator.lats[southernmost_lat_idx])

                if ice_edge_lats:
                    ice_edge_latitude = float(np.mean(ice_edge_lats))
                else:
                    ice_edge_latitude = 75.0
            else:
                ice_edge_latitude = 90.0

            # Расчет площади льда (примерный)
            ice_area = float(np.sum(ice_mask) * 25)  # 25 км² на ячейку (примерно)

            ForecastResult.objects.create(
                job=job,
                date=date,
                mean_ice_concentration=float(np.mean(predictions[0])),
                min_ice_concentration=float(np.min(predictions[0])),
                max_ice_concentration=float(np.max(predictions[0])),
                ice_area=ice_area,
                ice_edge_latitude=ice_edge_latitude,
                full_data={'prediction': predictions[0].tolist()}
            )

        job.status = 'completed'
        job.save()

        return {'success': True, 'job_id': job_id}

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in run_prediction: {error_trace}")

        job = PredictionJob.objects.get(job_id=job_id)
        job.status = 'failed'
        job.error_message = str(e)
        job.logs = error_trace
        job.save()
        return {'success': False, 'error': str(e)}


# predictor/views.py - замените функцию create_visualization

def create_visualization(job_id, input_data, output_data):
    """Обертка для безопасной визуализации"""
    from .ml_model.visualization import create_visualization_safe

    plot_path = create_visualization_safe(
        job_id,
        input_data,
        output_data,
        settings.MEDIA_ROOT / 'plots'
    )
    return plot_path

def results(request, job_id):
    """Страница с результатами"""
    job = get_object_or_404(PredictionJob, job_id=job_id)

    # Получаем прогнозы
    forecasts = job.results.all()

    # Конвертируем plot в base64 для отображения
    plot_base64 = None
    if job.plot_path and os.path.exists(job.plot_path):
        with open(job.plot_path, 'rb') as f:
            plot_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Подготавливаем данные для графиков
    dates = [f.date.strftime('%d.%m.%Y') for f in forecasts]
    mean_values = [f.mean_ice_concentration for f in forecasts]

    context = {
        'title': f'Результаты #{job_id}',
        'job': job,
        'forecasts': forecasts,
        'plot_base64': plot_base64,
        'dates_json': json.dumps(dates),
        'mean_values_json': json.dumps(mean_values),
        'active_page': 'history'
    }

    return render(request, 'predictor/results.html', context)


def visualize(request, job_id):
    """Страница с визуализацией"""
    job = get_object_or_404(PredictionJob, job_id=job_id)

    # Загружаем данные если есть
    input_data = None
    output_data = None

    if job.input_data_path and os.path.exists(job.input_data_path):
        input_data = np.load(job.input_data_path)
    if job.output_data_path and os.path.exists(job.output_data_path):
        output_data = np.load(job.output_data_path)

    context = {
        'title': f'Визуализация #{job_id}',
        'job': job,
        'input_data': input_data.tolist() if input_data is not None else None,
        'output_data': output_data.tolist() if output_data is not None else None,
        'active_page': 'history'
    }

    return render(request, 'predictor/visualize.html', context)


# predictor/views.py

def download_results(request, job_id, file_type):
    """Скачать результаты прогноза"""
    job = get_object_or_404(PredictionJob, job_id=job_id)

    if file_type == 'plot' and job.plot_path and os.path.exists(job.plot_path):
        return FileResponse(open(job.plot_path, 'rb'), as_attachment=True, filename=f'prediction_{job_id}.png')

    elif file_type == 'csv':
        # Создаем CSV с результатами
        import csv
        from io import StringIO

        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(['Date', 'Mean Ice Concentration', 'Min', 'Max', 'Ice Area', 'Ice Edge Latitude'])

        for f in job.results.all():
            writer.writerow([
                f.date,
                f.mean_ice_concentration,
                f.min_ice_concentration,
                f.max_ice_concentration,
                f.ice_area,
                f.ice_edge_latitude
            ])

        buffer.seek(0)
        response = HttpResponse(buffer, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename=prediction_{job_id}.csv'
        return response

    elif file_type == 'json':
        # Создаем JSON с результатами
        data = {
            'job_id': job.job_id,
            'created_at': job.created_at.isoformat(),
            'parameters': {
                'sequence_length': job.sequence_length,
                'forecast_horizon': job.forecast_horizon
            },
            'metrics': {
                'mse': job.mse,
                'mae': job.mae,
                'rmse': job.rmse
            },
            'forecasts': []
        }

        for f in job.results.all():
            data['forecasts'].append({
                'date': f.date.isoformat(),
                'mean_ice_concentration': f.mean_ice_concentration,
                'min_ice_concentration': f.min_ice_concentration,
                'max_ice_concentration': f.max_ice_concentration,
                'ice_area': f.ice_area,
                'ice_edge_latitude': f.ice_edge_latitude
            })

        response = JsonResponse(data)
        response['Content-Disposition'] = f'attachment; filename=prediction_{job_id}.json'
        return response

    else:
        messages.error(request, 'Неверный тип файла')
        return redirect('predictor:results', job_id=job_id)


def history(request):
    """История прогнозов"""
    jobs = PredictionJob.objects.all().order_by('-created_at')[:20]

    context = {
        'title': 'История прогнозов',
        'jobs': jobs,
        'active_page': 'history'
    }
    return render(request, 'predictor/history.html', context)


def about(request):
    """Страница 'О проекте'"""
    context = {
        'title': 'О проекте',
        'active_page': 'about'
    }
    return render(request, 'predictor/about.html', context)


def api_predict(request):
    """API для прогнозирования"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            sequence_length = data.get('sequence_length', 30)

            # Запускаем прогноз
            job_id = str(uuid.uuid4())[:8]
            job = PredictionJob.objects.create(
                job_id=job_id,
                sequence_length=sequence_length,
                status='processing'
            )

            # В реальном приложении здесь асинхронный запуск
            result = run_prediction(job_id, sequence_length, 30)

            if result['success']:
                return JsonResponse({
                    'success': True,
                    'job_id': job_id,
                    'status': 'completed'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': result['error']
                }, status=500)

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


logger = logging.getLogger(__name__)


@require_POST
@csrf_protect
def delete_job(request, job_id):
    """Удаление конкретного прогноза"""
    logger.info(f"Попытка удаления прогноза: {job_id}")

    try:
        # Получаем прогноз
        job = get_object_or_404(PredictionJob, job_id=job_id)
        logger.info(f"Прогноз найден: {job.job_id}")

        # Удаляем связанные результаты
        results_deleted = ForecastResult.objects.filter(job=job).delete()
        logger.info(f"Удалено результатов: {results_deleted[0] if results_deleted else 0}")

        # Запоминаем ID для сообщения
        job_id_deleted = job.job_id

        # Удаляем сам прогноз
        job.delete()
        logger.info(f"Прогноз {job_id_deleted} успешно удален")

        messages.success(request, f'✅ Прогноз {job_id_deleted} успешно удален')

    except PredictionJob.DoesNotExist:
        logger.error(f"Прогноз {job_id} не найден")
        messages.error(request, f'❌ Прогноз {job_id} не найден')

    except Exception as e:
        logger.error(f"Ошибка при удалении: {str(e)}")
        messages.error(request, f'❌ Ошибка при удалении: {str(e)}')

    return redirect('predictor:history')


@require_POST
@csrf_protect
def clear_history(request):
    """Очистка истории прогнозов"""
    logger.info("Попытка очистки истории")

    try:
        days = int(request.POST.get('days', 30))
        delete_all = request.POST.get('delete_all') == 'true'

        if delete_all:
            logger.info("Удаление ВСЕХ прогнозов")
            # Получаем все прогнозы
            all_jobs = PredictionJob.objects.all()
            count = all_jobs.count()

            # Удаляем связанные результаты
            for job in all_jobs:
                ForecastResult.objects.filter(job=job).delete()

            # Удаляем все прогнозы
            all_jobs.delete()
            logger.info(f"Удалено ВСЕХ прогнозов: {count}")
            messages.success(request, f'✅ Удалено ВСЕХ {count} прогнозов')

        else:
            logger.info(f"Удаление прогнозов старше {days} дней")
            # Удаляем прогнозы старше указанного количества дней
            threshold_date = timezone.now() - timedelta(days=days)
            old_jobs = PredictionJob.objects.filter(created_at__lt=threshold_date)

            count = old_jobs.count()
            logger.info(f"Найдено прогнозов для удаления: {count}")

            if count > 0:
                # Удаляем связанные результаты
                for job in old_jobs:
                    ForecastResult.objects.filter(job=job).delete()

                # Удаляем прогнозы
                old_jobs.delete()
                logger.info(f"Удалено {count} прогнозов")
                messages.success(request, f'✅ Удалено {count} прогнозов старше {days} дней')
            else:
                messages.info(request, f'ℹ️ Нет прогнозов старше {days} дней')

    except Exception as e:
        logger.error(f"Ошибка при очистке истории: {str(e)}")
        messages.error(request, f'❌ Ошибка: {str(e)}')

    return redirect('predictor:history')