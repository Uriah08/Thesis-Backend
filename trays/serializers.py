from rest_framework import serializers
from .models import TrayModel, TrayStepModel

class TraySerializer(serializers.ModelSerializer):
    farm_name = serializers.CharField(source='farm.name', read_only=True)
    session_name = serializers.CharField(source='session.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    created_by_profile_picture = serializers.CharField(source='created_by.profile_picture', read_only=True)

    class Meta:
        model = TrayModel
        fields = [
            'id',
            'farm',
            'farm_name',
            'name',
            'created_at',
            'finished_at',
            'session',
            'session_name',
            'created_by',
            'created_by_username',
            'created_by_profile_picture',
        ]
        read_only_fields = ['id','created_by', 'name', 'created_at', 'finished_at']

class TrayStepSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    created_by_profile_picture = serializers.CharField(source='created_by.profile_picture', read_only=True)
    class Meta:
        model = TrayStepModel
        fields = ['id', 'tray', 'title', 'description', 'image', 'datetime', 'created_by', 'created_by_username', 'created_by_profile_picture']
        read_only_fields = ['id', 'datetime', 'created_by',]
