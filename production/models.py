from django.db import models
from django.utils import timezone
from farms.models import FarmModel

# Create your models here.
class FarmProductionModel(models.Model):
    farm = models.ForeignKey(
        FarmModel,
        on_delete=models.CASCADE,
        related_name='farmproductions',
    )
    title = models.CharField(max_length=255)
    notes = models.TextField(blank=True, null=True)
    satisfaction = models.IntegerField(default=3)
    quantity = models.FloatField(default=0)
    landing = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.title} ({self.farm.name})"