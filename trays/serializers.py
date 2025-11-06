from rest_framework import serializers
from .models import SessionTrayModel, TrayStepModel

class SessionTraySerializer(serializers.ModelSerializer):
    farm_name = serializers.CharField(source='farm.name', read_only=True)
    session_name = serializers.CharField(source='session.name', read_only=True)
    tray_name = serializers.CharField(source='tray.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    created_by_profile_picture = serializers.CharField(source='created_by.profile_picture', read_only=True)

    class Meta:
        model = SessionTrayModel
        fields = [
            'id',
            'farm',
            'farm_name',
            'session',
            'session_name',
            'tray',
            'tray_name',
            'created_at',
            'finished_at',
            'created_by',
            'created_by_username',
            'created_by_profile_picture',
        ]
        read_only_fields = [
            'id',
            'created_by',
            'created_at',
            'finished_at',
            'farm_name',
            'session_name',
            'tray_name',
            'created_by_username',
            'created_by_profile_picture',
        ]

class TrayStepSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    created_by_profile_picture = serializers.CharField(source='created_by.profile_picture', read_only=True)
    class Meta:
        model = TrayStepModel
        fields = ['id', 'tray', 'title', 'description', 'image', 'datetime', 'created_by', 'created_by_username', 'created_by_profile_picture']
        read_only_fields = ['id', 'datetime', 'created_by',]
