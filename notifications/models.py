from django.conf import settings
from django.db import models
from django.utils.timezone import now

class DeviceToken(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="device_tokens",
        null=True,
        blank=True
    )
    token = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "token")

    def __str__(self):
        return f"{self.user.username if self.user else 'NoUser'} - {self.token}"
    
class Notification(models.Model):
    NOTIFICATION_TYPES = [
        ('announcement', 'Announcement'),
        ('reminder', 'Reminder'),
        ('weather', 'Weather'),
        ('people', 'People'),
    ]
    title = models.CharField(max_length=255)
    type = models.CharField(max_length=100, choices=NOTIFICATION_TYPES) 
    body = models.TextField()
    data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} ({self.type})"

class Recipient(models.Model):
    notification = models.ForeignKey(
        Notification,
        on_delete=models.CASCADE,
        related_name="recipients"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notifications"
    )
    read = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("notification", "user")

    def mark_as_read(self):
        if not self.read:
            self.read = True
            self.read_at = now()
            self.save()

    def __str__(self):
        return f"{self.user.username} - {self.notification.title} - {'Read' if self.read else 'Unread'}"