# predictor/templatetags/predictor_tags.py

from django import template
from predictor.models import PredictionJob

register = template.Library()

@register.simple_tag
def get_total_jobs():
    return PredictionJob.objects.count()

@register.simple_tag
def get_oldest_job():
    return PredictionJob.objects.order_by('created_at').first()

@register.simple_tag
def get_newest_job():
    return PredictionJob.objects.order_by('-created_at').first()