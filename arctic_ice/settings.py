# arctic_ice/settings.py

import os
from pathlib import Path

# Базовая директория
BASE_DIR = Path(__file__).resolve().parent.parent

# Секретный ключ (в продакшне должен быть в переменных окружения)
SECRET_KEY = 'django-insecure-your-secret-key-here'

# Режим отладки
DEBUG = True

# settings.py - только для локальной сети!
ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '.ngrok.io',
    '.ngrok-free.app',
    'yee-expurgatory-inadvertently.ngrok-free.dev',  # ваш точный адрес
]

DEBUG = False # можно оставить True

# Установленные приложения
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_bootstrap5',
    'predictor',  # наше приложение
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'arctic_ice.urls'

# Шаблоны
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'arctic_ice.wsgi.application'

# База данных (используем SQLite для простоты)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Валидация паролей
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Интернационализация
LANGUAGE_CODE = 'ru-ru'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Статические файлы (CSS, JS, изображения)
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'predictor', 'static'),
]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Медиа файлы (загруженные пользователями)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Настройки для моделей машинного обучения
ML_MODELS_PATH = BASE_DIR / 'predictor' / 'ml_model' / 'trained_models'
os.makedirs(ML_MODELS_PATH, exist_ok=True)

# Максимальный размер загружаемых файлов (10MB)
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

# Настройки по умолчанию для полей
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# В самом КОНЦЕ файла добавьте:
import mimetypes
mimetypes.add_type("text/css", ".css", True)

# Для ngrok добавьте:
CSRF_TRUSTED_ORIGINS = [
    'https://*.ngrok-free.app',
    'https://*.ngrok.io',
    'https://yee-expurgatory-inadvertently.ngrok-free.dev',
]

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
MIDDLEWARE.append('predictor.middleware.ExceptionLoggingMiddleware')