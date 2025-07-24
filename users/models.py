from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    address = models.TextField(blank=True)
    birthday = models.DateField(null=True, blank=True)
    is_complete = models.BooleanField(default=False)
    profile_picture = models.URLField(null=True, blank=True)

    def __str__(self):
        return self.username
