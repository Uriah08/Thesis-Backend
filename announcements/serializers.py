from rest_framework import serializers
from .models import AnnouncementModel

class AnnouncementSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    created_by_profile_picture = serializers.CharField(source='created_by.profile_picture', read_only=True)
    
    class Meta:
        model = AnnouncementModel
        fields = [
            "id",
            "farm",
            "title",
            "content",
            "status",
            "created_by",
            "created_at",
            "expires_at",
            'created_by_username',
            'created_by_profile_picture',
        ]
        read_only_fields = ["id", "created_at", 'created_by_username', 'created_by_profile_picture']