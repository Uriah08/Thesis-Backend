from rest_framework import serializers
from .models import FarmTrayModel

class FarmTraySerializer(serializers.ModelSerializer):
    farm_name = serializers.CharField(source="farm.name", read_only=True)

    class Meta:
        model = FarmTrayModel
        fields = ["id", "farm", "farm_name", "name", "description", "status", "created_at"]
        read_only_fields = ["status", "created_at"]
