@echo off
echo ========================================
echo ОЧИСТКА ИСТОРИИ ПРОГНОЗОВ
echo ========================================

cd /d C:\Users\Valen\PycharmProjects\NeuralNetworkforCreatingGeographicModels\arctic_ice
call .venv\Scripts\activate

echo Удаляем прогнозы старше 45 дней...
python manage.py cleanup_predictions --days 45 --keep-last 100

echo ========================================
echo Очистка завершена!
pause