from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from django.utils import timezone
from trays.models import SessionTrayModel

from .models import FarmSessionModel
from .serializers import FarmSessionSerializer

class CreateFarmSessionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FarmSessionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class GetFarmSessionsView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, farm_id):
        if not farm_id:
            return Response({"detail": "No Farm ID provided."}, status=status.HTTP_400_BAD_REQUEST)
        
        sessions = FarmSessionModel.objects.filter(farm_id=farm_id)
        serializer = FarmSessionSerializer(sessions, many=True)
        
        return Response(serializer.data, status=status.HTTP_200_OK)

    
class GetSessionByIdView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, id):
        try:
            session = FarmSessionModel.objects.get(id=id)
            serializer = FarmSessionSerializer(session)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except FarmSessionModel.DoesNotExist:
            return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)

class ActivateFarmSessionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request, id):
        try:
            if not id:
                return Response(
                    {"detail": "No Session ID provided."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            session = FarmSessionModel.objects.get(id=id)

            if session.status == 'inactive':
                session.status = 'active'
                session.start_time = timezone.now()
                session.save()
                return Response(
                    {"detail": "Session activated successfully."}, 
                    status=status.HTTP_200_OK
                )

            elif session.status == 'active':
                session.status = 'finished'
                session.end_time = timezone.now()
                session.save()

                related_trays = SessionTrayModel.objects.filter(session=session)
                for tray in related_trays:
                    tray.finished_at = timezone.now()
                    tray.save()

                    if hasattr(tray.tray, "status"):
                        tray.tray.status = "inactive"
                        tray.tray.save()

                return Response(
                    {"detail": "Session finished and trays set to inactive."}, 
                    status=status.HTTP_200_OK
                )

            else:
                return Response(
                    {"detail": "Activation failed. Invalid session state."},
                    status=status.HTTP_400_BAD_REQUEST
                )

        except FarmSessionModel.DoesNotExist:
            return Response(
                {"detail": "No sessions available to activate."}, 
                status=status.HTTP_404_NOT_FOUND
            )

        
class RenameSessionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def patch(self, request, session_id):
        try:
            session = FarmSessionModel.objects.get(id=session_id)
        except FarmSessionModel.DoesNotExist:
            return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)

        new_name = request.data.get("name")
        if not new_name:
            return Response({"detail": "No name provided."}, status=status.HTTP_400_BAD_REQUEST)

        session.name = new_name
        session.save()

        serializer = FarmSessionSerializer(session)
        return Response(serializer.data, status=status.HTTP_200_OK)


class DeleteSessionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def delete(self, request, session_id):
        try:
            session = FarmSessionModel.objects.get(id=session_id)

            if session.status == "active":
                return Response(
                    {"detail": "Cannot delete an active session."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            session.delete()
            return Response(
                {"detail": "Session deleted successfully."},
                status=status.HTTP_200_OK
            )

        except FarmSessionModel.DoesNotExist:
            return Response(
                {"detail": "Session not found."},
                status=status.HTTP_404_NOT_FOUND
            )

