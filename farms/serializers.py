from rest_framework import serializers
from .models import FarmModel
from django.contrib.auth import get_user_model

User = get_user_model()

class FarmSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.id')
    members = serializers.PrimaryKeyRelatedField(
        many=True,
        queryset=User.objects.all(),
        required=False
    )
    password = serializers.CharField(write_only=True)
    
    class Meta:
        model = FarmModel
        fields = [
            'id',
            'name',
            'description',
            'image_url',
            'password',
            'owner',
            'members',
        ]
    
    def create(self, validated_data):
        user = self.context['request'].user
        members = validated_data.pop('members', [])
        farm = FarmModel.objects.create(owner=user, **validated_data)
    
        farm.members.add(user, *members)
        return farm

class JoinFarmSerializer(serializers.Serializer):
    farm_id = serializers.IntegerField()
    password = serializers.CharField()    
    