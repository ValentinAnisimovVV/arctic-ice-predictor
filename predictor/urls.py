# predictor/urls.py
from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('results/<str:job_id>/', views.results, name='results'),
    path('visualize/<str:job_id>/', views.visualize, name='visualize'),
    path('about/', views.about, name='about'),
    path('history/', views.history, name='history'),
    path('clear-history/', views.clear_history, name='clear_history'),
    path('delete-job/<str:job_id>/', views.delete_job, name='delete_job'),  # ЭТОТ URL
    path('download/<str:job_id>/<str:file_type>/', views.download_results, name='download'),
    path('api/predict/', views.api_predict, name='api_predict'),
]