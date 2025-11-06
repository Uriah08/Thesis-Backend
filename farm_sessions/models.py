from django.db import models
from django.utils import timezone
from farms.models import FarmModel

class FarmSessionModel(models.Model):
    farm = models.ForeignKey(
        FarmModel,
        on_delete=models.CASCADE,
        related_name='sessions',
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    status = models.CharField(default='inactive', max_length=50)
    start_time = models.DateTimeField(blank=True, null=True)
    end_time = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.name} ({self.farm.name})"
    
