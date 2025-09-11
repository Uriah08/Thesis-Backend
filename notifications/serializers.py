from rest_framework import serializers
from .models import DeviceToken, Notification, Recipient
from django.contrib.auth import get_user_model
from django.utils.timezone import now

User = get_user_model()

class DeviceTokenSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceToken
        fields = ["token"]

class NotificationSerializer(serializers.ModelSerializer):
    recipients = serializers.StringRelatedField(many=True, read_only=True)
    
    class Meta:
        model = Notification
        fields = [
            "id",
            "title",
            "type",
            "body",
            "data",
            "created_at",
            "updated_at",
            "recipients",
        ]

class RecipientSerializer(serializers.ModelSerializer):
    notification = NotificationSerializer(read_only=True)
    user = serializers.StringRelatedField()

    class Meta:
        model = Recipient
        fields = [
            "id",
            "notification",
            "user",
            "read",
            "read_at",
            "created_at",
        ]