from rest_framework import status, permissions
from rest_framework.authentication import TokenAuthentication
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import SessionTrayModel, TrayStepModel
from .serializers import SessionTraySerializer, TrayStepSerializer
from django.utils import timezone
from farms.models import FarmModel
from farm_sessions.models import FarmSessionModel
from farm_trays.models import FarmTrayModel

class TrayCreateView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]

    def post(self, request):
        tray_ids = request.data.get("tray_ids", [])
        farm_id = request.data.get("farm")
        session_id = request.data.get("session")

        if not tray_ids or not isinstance(tray_ids, list):
            return Response({"error": "tray_ids must be a non-empty list."}, status=status.HTTP_400_BAD_REQUEST)
        if not farm_id or not session_id:
            return Response({"error": "farm and session are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            farm = FarmModel.objects.get(id=farm_id)
            session = FarmSessionModel.objects.get(id=session_id)
        except (FarmModel.DoesNotExist, FarmSessionModel.DoesNotExist):
            return Response({"error": "Invalid farm or session ID."}, status=status.HTTP_404_NOT_FOUND)

        created_trays = []

        for tray_id in tray_ids:
            try:
                tray = FarmTrayModel.objects.get(id=tray_id)
            except FarmTrayModel.DoesNotExist:
                continue  
            
            tray.status = "active"
            tray.save(update_fields=["status"])

            tray_count = SessionTrayModel.objects.filter(session=session).count()
            next_name = f"Tray {tray_count + 1}"

            session_tray = SessionTrayModel.objects.create(
                farm=farm,
                session=session,
                tray=tray,
                created_by=request.user,
            )

            created_trays.append(session_tray)

        serializer = SessionTraySerializer(created_trays, many=True)

        return Response(
            {
                "message": f"{len(created_trays)} trays added successfully to session '{session.name}'.",
                "data": serializer.data,
            },
            status=status.HTTP_201_CREATED,
        )
    
class GetTraysView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]

    def get(self, request, session_id):
        trays = SessionTrayModel.objects.filter(session__id=session_id)
        serializer = SessionTraySerializer(trays, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
class GetTrayView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]
    
    def get(self, request, id):
        try:
            tray = SessionTrayModel.objects.get(id=id)
            serializer = SessionTraySerializer(tray)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except SessionTrayModel.DoesNotExist:
            return Response({"error": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)
        
class CreateTrayProgressView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]
    
    def post(self, request):
        tray_id = request.data.get("tray")
        
        if not tray_id:
            return Response({"error": "Tray ID is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            tray = SessionTrayModel.objects.get(id=tray_id)
        except SessionTrayModel.DoesNotExist:
            return Response({"error": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)
        
        if tray.finished_at:
            return Response(
                {"error": "Cannot add progress to a harvested tray."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        serializer = TrayStepSerializer(data=request.data)
        if serializer.is_valid():
            tray_step = serializer.save(
                tray=tray,
                created_by=request.user
            )
            response_data = TrayStepSerializer(tray_step).data
            response_data["message"] = f"Progress step '{tray_step.title}' added successfully."
            return Response(response_data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GetTrayProgressView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]

    def get(self, request, tray_id):
        try:
            tray = SessionTrayModel.objects.get(id=tray_id)
        except SessionTrayModel.DoesNotExist:
            return Response({"error": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)

        steps = TrayStepModel.objects.filter(tray=tray).order_by("datetime")
        serializer = TrayStepSerializer(steps, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

class HarvestTrayView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]
    
    def post(self, request, tray_id):
        try:
            tray = SessionTrayModel.objects.get(id=tray_id)
        except SessionTrayModel.DoesNotExist:
            return Response({"error": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)
        
        if tray.finished_at:
            return Response({"message": "Tray already harvested."}, status=status.HTTP_400_BAD_REQUEST)

        tray.finished_at = timezone.now()
        tray.save()

        tray.tray.status = "inactive"
        tray.tray.save()

        serializer = SessionTraySerializer(tray)
        return Response({
            "message": "Tray successfully harvested and farm tray set to inactive.",
            "tray": serializer.data
        }, status=status.HTTP_200_OK)

class DeleteTrayView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]
    
    def delete(self, request, tray_id):
        try:
            tray = SessionTrayModel.objects.get(id=tray_id)
        except SessionTrayModel.DoesNotExist:
            return Response({"error": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)
        
        if tray.created_by != request.user:
            return Response({"error": "You do not have permission to delete this tray."}, status=status.HTTP_403_FORBIDDEN)
        
        if tray.finished_at is None:
            farm_tray = tray.tray
            if farm_tray.status != "inactive":
                farm_tray.status = "inactive"
                farm_tray.save()

        tray.delete()

        return Response(
            {"message": "Tray deleted successfully."},
            status=status.HTTP_200_OK
        )