from django.db import models
from django.conf import settings
from django.utils import timezone

class FarmModel(models.Model):
    name = models.CharField(max_length=132)
    description = models.TextField(blank=True, null=True)
    image_url = models.URLField(blank=True, null=True)
    password = models.CharField(max_length=128)
    create_at = models.DateField(default=timezone.now)
    
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='owned_farms',
        on_delete=models.CASCADE
    )
    
    members = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name='member_farms',
        blank=True
    )
    
    def __str__(self):
        return self.name
    