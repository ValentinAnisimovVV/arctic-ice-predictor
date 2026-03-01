# predictor/management/commands/cleanup_predictions.py

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from predictor.models import PredictionJob, ForecastResult
from django.db import transaction
from django.conf import settings


class Command(BaseCommand):
    help = 'Очищает старые прогнозы из базы данных'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=45,
            help='Сколько дней хранить прогнозы (по умолчанию 45)'
        )
        parser.add_argument(
            '--keep-last',
            type=int,
            default=100,
            help='Сколько последних записей сохранять (по умолчанию 100)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Показать, что будет удалено, но не удалять'
        )

    def handle(self, *args, **options):
        days = options['days']
        keep_last = options['keep_last']
        dry_run = options['dry_run']

        self.stdout.write(self.style.WARNING(f"🔍 Режим: {'ПРОСМОТР (dry-run)' if dry_run else 'УДАЛЕНИЕ'}"))
        self.stdout.write(f"📅 Храним прогнозы: {days} дней")
        self.stdout.write(f"📊 Храним последние: {keep_last} записей")

        # Вариант 1: Удаление по дате (старше N дней)
        threshold_date = timezone.now() - timedelta(days=days)
        old_jobs_by_date = PredictionJob.objects.filter(
            created_at__lt=threshold_date
        )

        count_by_date = old_jobs_by_date.count()

        if dry_run:
            self.stdout.write(f"📋 Будет удалено {count_by_date} прогнозов старше {days} дней")
            if count_by_date > 0:
                self.stdout.write("   Примеры:")
                for job in old_jobs_by_date[:5]:
                    self.stdout.write(f"   - {job.job_id}: {job.created_at.strftime('%Y-%m-%d %H:%M')}")
        else:
            # Сначала удаляем связанные ForecastResult
            for job in old_jobs_by_date:
                ForecastResult.objects.filter(job=job).delete()
            # Затем удаляем сами прогнозы
            old_jobs_by_date.delete()
            self.stdout.write(self.style.SUCCESS(f"✅ Удалено {count_by_date} прогнозов старше {days} дней"))

        # Вариант 2: Оставить только последние N записей (если включено)
        if keep_last > 0 and not dry_run:
            total_count = PredictionJob.objects.count()
            if total_count > keep_last:
                to_delete = total_count - keep_last
                # Получаем самые старые записи для удаления
                oldest = PredictionJob.objects.order_by('created_at')[:to_delete]

                # Сначала удаляем связанные результаты
                for job in oldest:
                    ForecastResult.objects.filter(job=job).delete()

                # Затем удаляем прогнозы
                oldest.delete()
                self.stdout.write(self.style.SUCCESS(f"✅ Оставлено последних {keep_last} записей, удалено {to_delete}"))

        if dry_run:
            self.stdout.write(self.style.WARNING("\n💡 Запустите без --dry-run для реального удаления"))
        else:
            self.stdout.write(self.style.SUCCESS("🎉 Очистка завершена!"))