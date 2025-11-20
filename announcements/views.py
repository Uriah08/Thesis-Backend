from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .models import AnnouncementModel
from .serializers import AnnouncementSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from notifications.models import DeviceToken, Notification, Recipient
from core.expo import send_push_notification
from farms.models import FarmModel


# Create your views here.

class CreateAnnouncementView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        data = request.data.copy()
        data["created_by"] = request.user.id
        
        serializer = AnnouncementSerializer(data=data)
        if serializer.is_valid():
            announcement = serializer.save()
            
            farm_id = data.get("farm")
            
            try:
                farm = FarmModel.objects.get(id=farm_id)
            except FarmModel.DoesNotExist:
                return Response({"detail": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)
            
            members = farm.members.exclude(id=request.user.id)
            
            notification_data = {
                "title": f"{announcement.title}",
                "type": "announcement",
                "body": announcement.content,
                "data": {
                    "announcement_id": announcement.id,
                    "farm_id": farm.id,
                    "created_by": request.user.id,
                    "expires_at": announcement.expires_at.isoformat() if announcement.expires_at else None
                }
            }
            
            notification = Notification.objects.create(**notification_data)
            
            recipients = [Recipient(notification=notification, user=user) for user in members]
            Recipient.objects.bulk_create(recipients)
            
            device_tokens = DeviceToken.objects.filter(user__in=members)
            
            for token in device_tokens:
                send_push_notification(
                    token.token,
                    announcement.title,
                    announcement.content
                )
            
            return Response(
                {"detail": "Announcement created successfully", "data": serializer.data},
                status=status.HTTP_201_CREATED
            )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class GetAnnouncementView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, id):
        announcements = AnnouncementModel.objects.filter(farm_id=id).order_by('-created_at')
        serializer = AnnouncementSerializer(announcements, many=True)
        return Response(serializer.data)

class DeleteAnnouncementView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def delete(self, request, id):
        try:
            announcement = AnnouncementModel.objects.get(id=id)
        except AnnouncementModel.DoesNotExist:
            return Response({"detail": "Announcement not found."}, status=status.HTTP_404_NOT_FOUND)
        
        announcement.delete()
        return Response({"detail": "Announcement deleted successfully."}, status=status.HTTP_200_OK)