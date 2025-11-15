from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .models import AnnouncementModel
from .serializers import AnnouncementSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication

# Create your views here.

class CreateAnnouncementView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        data = request.data.copy()
        data["created_by"] = request.user.id
        
        serializer = AnnouncementSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
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