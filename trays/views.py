from rest_framework import status, permissions
from rest_framework.authentication import TokenAuthentication
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import TrayModel, TrayStepModel
from .serializers import TraySerializer, TrayStepSerializer

class TrayCreateView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]

    def post(self, request):
        serializer = TraySerializer(data=request.data)
        if serializer.is_valid():
            farm = serializer.validated_data.get("farm")
            session = serializer.validated_data.get("session")
            
            tray_count = TrayModel.objects.filter(session=session).count()
            next_name = f"Tray {tray_count + 1}"
            
            tray = serializer.save(
                created_by=request.user,
                name=next_name
            )

            response_data = TraySerializer(tray).data
            response_data["message"] = f"{next_name} created successfully."
            
            return Response(response_data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class GetTraysView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]

    def get(self, request, session_id):
        trays = TrayModel.objects.filter(session__id=session_id)
        serializer = TraySerializer(trays, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
class GetTrayView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]
    
    def get(self, request, id):
        try:
            tray = TrayModel.objects.get(id=id)
            serializer = TraySerializer(tray)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except TrayModel.DoesNotExist:
            return Response({"error": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)
        
class CreateTrayProgressView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    authentication_classes = [TokenAuthentication]
    
    def post(self, request):
        tray_id = request.data.get("tray")
        
        if not tray_id:
            return Response({"error": "Tray ID is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            tray = TrayModel.objects.get(id=tray_id)
        except TrayModel.DoesNotExist:
            return Response({"error": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)
        
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
            tray = TrayModel.objects.get(id=tray_id)
        except TrayModel.DoesNotExist:
            return Response({"error": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)

        steps = TrayStepModel.objects.filter(tray=tray).order_by("datetime")
        serializer = TrayStepSerializer(steps, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)
