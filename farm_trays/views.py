from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from .serializers import FarmTraySerializer

from .models import FarmTrayModel

class CreateFarmTrayView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FarmTraySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {"detail": "Tray created successfully.", "data": serializer.data},
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GetFarmTraysView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, farm_id):
        trays = FarmTrayModel.objects.filter(farm_id=farm_id).order_by('-created_at')
        serializer = FarmTraySerializer(trays, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
class GetFarmTrayByIdView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, tray_id):
        try:
            tray = FarmTrayModel.objects.get(id=tray_id)
            serializer = FarmTraySerializer(tray)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except FarmTrayModel.DoesNotExist:
            return Response({"detail": "Tray not found."}, status=status.HTTP_404_NOT_FOUND)
    