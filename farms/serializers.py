from rest_framework import serializers
from .models import FarmModel
from django.contrib.auth import get_user_model
from django.utils import timezone
from farm_trays.models import FarmTrayModel
from farm_sessions.models import FarmSessionModel
from trays.models import SessionTrayModel, TrayStepModel
from announcements.models import AnnouncementModel
from django.contrib.auth import get_user_model
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate

User = get_user_model()

class FarmSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.id')
    owner_name = serializers.ReadOnlyField(source='owner.username')
    members = serializers.PrimaryKeyRelatedField(
        many=True,
        queryset=User.objects.all(),
        required=False
    )
    blocked = serializers.PrimaryKeyRelatedField(
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
            'owner_name',
            'members',
            'blocked',
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
    
class MemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'first_name', 'last_name', 'username', 'email', 'profile_picture']

class FarmDashboardSerializer(serializers.ModelSerializer):
    tray_count = serializers.SerializerMethodField()
    announcement_count = serializers.SerializerMethodField()
    session_trays_count_by_day = serializers.SerializerMethodField()
    detected_and_reject_by_day = serializers.SerializerMethodField()
    recent_harvested_trays = serializers.SerializerMethodField()

    class Meta:
        model = FarmModel
        fields = [
            'id', 
            'name', 
            'description', 
            'image_url', 
            'session_trays_count_by_day', 
            'tray_count', 
            'announcement_count', 
            'detected_and_reject_by_day',
            'recent_harvested_trays'
        ]
    
    def get_tray_count(self, obj):
        return FarmTrayModel.objects.filter(farm=obj).count()
    
    def get_announcement_count(self, obj):
        return AnnouncementModel.objects.filter(farm=obj).count()
    
    # ---------------------------------------------
    # FIXED: use tray__farm instead of farm
    # ---------------------------------------------
    def get_session_trays_count_by_day(self, obj):
        session_trays = (
            SessionTrayModel.objects
            .filter(tray__farm=obj, finished_at__isnull=False)
            .annotate(day=TruncDate('finished_at'))
            .values('day')
            .annotate(count=Count('id'))
            .order_by('day')
        )
        
        return [{'finished_at': t['day'], 'count': t['count']} for t in session_trays]

    # ---------------------------------------------
    # FIXED: use tray__farm instead of farm
    # ---------------------------------------------
    def get_detected_and_reject_by_day(self, obj):
        daily_stats = (
            SessionTrayModel.objects
            .filter(tray__farm=obj, created_at__isnull=False)
            .annotate(day=TruncDate('created_at'))
            .values('day')
            .annotate(
                total_detected=Sum('steps__detected'),
                total_rejects=Sum('steps__rejects')
            )
            .order_by('day')
        )

        return [
            {
                'day': stat['day'],
                'detected': stat['total_detected'] or 0,
                'rejects': stat['total_rejects'] or 0
            }
            for stat in daily_stats
        ]

    # ---------------------------------------------
    # FIXED: use tray__farm instead of farm
    # ---------------------------------------------
    def get_recent_harvested_trays(self, obj):
        trays = (
            SessionTrayModel.objects
            .filter(tray__farm=obj, finished_at__isnull=False)
            .order_by('-finished_at')[:3]
        )

        return [
            {
                "id": tray.id,
                "finished_at": tray.finished_at,
                "created_at": tray.created_at,
                "tray_name": tray.tray.name if tray.tray else None,
                "tray_id": tray.tray.id if tray.tray else None,
            }
            for tray in trays
        ]
    
