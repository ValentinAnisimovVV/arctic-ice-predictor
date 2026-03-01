#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Фикс на уровне manage.py

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'arctic_ice.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()