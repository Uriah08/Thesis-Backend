from rest_framework import serializers
from .models import FarmSessionModel

class FarmSessionSerializer(serializers.ModelSerializer):
    farm_name = serializers.CharField(source="farm.name", read_only=True)
    class Meta:
        model = FarmSessionModel
        fields = [
            "id",
            "farm",
            "farm_name",
            "name",
            "description",
            "status",
            "start_time",
            "end_time",
            "created_at",
        ]
        read_only_fields = ["created_at"]