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
    path('download/<str:job_id>/<str:file_type>/', views.download_results, name='download'),
    path('history/', views.history, name='history'),
    path('api/predict/', views.api_predict, name='api_predict'),
]