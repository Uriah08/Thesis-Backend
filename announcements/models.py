from django.db import models
from django.conf import settings
from farms.models import FarmModel

# Create your models here.
class AnnouncementModel(models.Model):
    farm = models.ForeignKey(
        FarmModel,
        related_name='announcements',
        on_delete=models.CASCADE
    )
    title = models.CharField(max_length=255)
    content = models.TextField()
    status = models.CharField(default='active', max_length=50)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='created_announcements'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"{self.title} ({self.farm.name})"