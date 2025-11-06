from rest_framework import serializers
from .models import FarmSessionModel

class FarmSessionSerializer(serializers.ModelSerializer):
    farm_name = serializers.CharField(source="farm.name", read_only=True)
    trays_count = serializers.SerializerMethodField()
    
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
            "trays_count"
        ]
        read_only_fields = ["created_at"]
    
    def get_trays_count(self, obj):
        return obj.session_trays.count()

