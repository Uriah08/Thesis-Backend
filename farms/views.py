from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication

from .models import FarmModel
from .serializers import FarmSerializer, JoinFarmSerializer

class CreateFarmView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FarmSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            serializer.save()
            return Response({"detail": "Farm created successfully."}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class JoinFarmView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = JoinFarmSerializer(data=request.data)
        if serializer.is_valid():
            farm_id = serializer.validated_data['farm_id']
            password = serializer.validated_data['password']

            try:
                farm = FarmModel.objects.get(id=farm_id)
            except FarmModel.DoesNotExist:
                return Response({"detail": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)

            if farm.password != password:
                return Response({"detail": "Incorrect password."}, status=status.HTTP_400_BAD_REQUEST)

            if request.user in farm.members.all():
                return Response({"detail": "You are already a member of this farm."}, status=status.HTTP_200_OK)

            farm.members.add(request.user)
            return Response({"detail": "Joined the farm successfully."}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ListUserFarmsView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        farms = FarmModel.objects.filter(members=request.user)
        serializer = FarmSerializer(farms, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
class GetFarmView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, id):
        try:
            farm = FarmModel.objects.get(id=id)
        except FarmModel.DoesNotExist:
            return Response({"detail": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = FarmSerializer(farm)
        return Response(serializer.data, status=status.HTTP_200_OK)