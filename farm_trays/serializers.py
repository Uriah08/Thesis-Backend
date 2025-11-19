from rest_framework import serializers
from .models import FarmTrayModel
from trays.models import SessionTrayModel
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate

class FarmTraySerializer(serializers.ModelSerializer):
    farm_name = serializers.CharField(source="farm.name", read_only=True)

    class Meta:
        model = FarmTrayModel
        fields = ["id", "farm", "farm_name", "name", "description", "status", "created_at"]
        read_only_fields = ["status", "created_at"]

class TrayDashboardSerializer(serializers.ModelSerializer):
    session_tray_count = serializers.SerializerMethodField()
    detected_and_reject_by_day = serializers.SerializerMethodField()
    recent_harvested_trays = serializers.SerializerMethodField()
    
    class Meta:
        model = FarmTrayModel
        fields = ["id", "name", "status", "created_at", "session_tray_count", "detected_and_reject_by_day", "recent_harvested_trays"]
        
    def get_session_tray_count(self, obj):
        session_trays = (
            SessionTrayModel.objects
            .filter(tray=obj, finished_at__isnull=False)
            .annotate(day=TruncDate('finished_at'))
            .values('day')
            .annotate(count=Count('id'))
            .order_by('day')
        )

        return [
            {
                'finished_at': t['day'],
                'count': t['count']
            }
            for t in session_trays
        ]
    
    def get_detected_and_reject_by_day(self, obj):
        daily_stats = (
            SessionTrayModel.objects
            .filter(tray=obj, farm=obj.farm, created_at__isnull=False)
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
    
    def get_recent_harvested_trays(self, obj):
        trays = (
            SessionTrayModel.objects
            .filter(tray=obj, farm=obj.farm, finished_at__isnull=False)
            .order_by('-finished_at')[:3]
        )

        return [
            {
                "id": tray.id,
                "finished_at": tray.finished_at,
                "created_at": tray.created_at,
                "tray_name": tray.tray.name if tray.tray else None,
                "session_name": tray.session.name if tray.session else None,
                "tray_id": tray.tray.id if tray.tray else None,
                "session_id": tray.session.id if tray.session else None,
            }
            for tray in trays
        ]
