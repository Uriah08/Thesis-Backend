from rest_framework import serializers
from .models import DeviceToken
from django.contrib.auth import get_user_model

User = get_user_model()

class DeviceTokenSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceToken
        fields = ["token"]

