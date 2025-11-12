from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication

from .models import FarmModel
from .serializers import FarmSerializer, JoinFarmSerializer, MemberSerializer

class CreateFarmView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FarmSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            farm = serializer.save()
            response_serializer = FarmSerializer(farm, context={"request": request})
            return Response({"detail": "Farm created successfully.","farm": response_serializer.data}, status=status.HTTP_201_CREATED)
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
    
class GetMembersView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, id):
        try:
            farm = FarmModel.objects.get(id=id)
        except FarmModel.DoesNotExist:
            return Response(
                {"detail": "Farm not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        members = farm.members.all()
        serializer = MemberSerializer(members, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
class EditFarmView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def patch(self, request):
        farm_id = request.data.get("id")

        if not farm_id:
            return Response(
                {"detail": "Farm ID is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            farm = FarmModel.objects.get(id=farm_id)
        except FarmModel.DoesNotExist:
            return Response(
                {"detail": "Farm not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        if farm.owner != request.user:
            return Response(
                {"detail": "You do not have permission to edit this farm."},
                status=status.HTTP_403_FORBIDDEN
            )

        serializer = FarmSerializer(farm, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {"detail": "Farm updated successfully.", "farm": serializer.data},
                status=status.HTTP_200_OK
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class FarmChangePassword(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def patch(self, request):
        farm_id = request.data.get("id")
        
        if not farm_id:
            return Response({"detail": "Farm ID is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            farm = FarmModel.objects.get(id=farm_id)
        except FarmModel.DoesNotExist:
            return Response({"detail": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)
        
        if farm.owner != request.user:
            return Response({"detail": "You do not have permission to edit this farm."}, status=status.HTTP_403_FORBIDDEN)
        
        new_password = request.data.get("new_password")
        old_password = request.data.get("old_password")
        confirm_password = request.data.get("confirm_password")
        
        if not new_password or not old_password or not confirm_password:
            return Response({"detail": "All fields are required."}, status=status.HTTP_400_BAD_REQUEST)
        
        if new_password != confirm_password:
            return Response({"detail": "Passwords do not match."}, status=status.HTTP_400_BAD_REQUEST)
        
        if farm.password != old_password:
            return Response({"detail": "Incorrect old password."}, status=status.HTTP_400_BAD_REQUEST)
        
        farm.password = new_password
        farm.save()
        
        return Response({"detail": "Password changed successfully."}, status=status.HTTP_200_OK)