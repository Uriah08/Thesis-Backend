from django.db import models
from farms.models import FarmModel
from farm_sessions.models import FarmSessionModel
from django.conf import settings
from farm_trays.models import FarmTrayModel

# Create your models here.
class SessionTrayModel(models.Model):
    farm = models.ForeignKey(
        FarmModel,
        on_delete=models.CASCADE,
        related_name='session_trays'
    )
    session = models.ForeignKey(
        FarmSessionModel,
        on_delete=models.CASCADE,
        related_name='session_trays'
    )
    tray = models.ForeignKey(
        FarmTrayModel,
        on_delete=models.CASCADE,
        related_name='session_trays'
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='created_session_trays',
        on_delete=models.CASCADE
    )
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"Session {self.session.name} - Tray {self.tray.name}"
    
class TrayStepModel(models.Model):
    tray = models.ForeignKey(
        SessionTrayModel,
        on_delete=models.CASCADE,
        related_name='steps'
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='tray_steps',
    )
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    image = models.TextField(blank=True, null=True)
    datetime = models.DateTimeField(auto_now_add=True)
    rejects = models.IntegerField(blank=True, null=True)
    detected = models.IntegerField(blank=True, null=True)

    class Meta:
        ordering = ['datetime']

    def __str__(self):
        return f"{self.tray.name} - {self.title}"