from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from django.contrib.auth import get_user_model

from .models import FarmModel
from .serializers import FarmSerializer, JoinFarmSerializer, MemberSerializer

User = get_user_model()


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

            if request.user in farm.blocked.all():
                return Response({"detail": "You are blocked from joining this farm."}, status=status.HTTP_403_FORBIDDEN)

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
    
class BlockUserFarmView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        farm_id = request.data.get("farm")
        user_ids = request.data.get("user_ids", [])

        if not farm_id or not user_ids:
            return Response(
                {"detail": "Farm ID and user_ids are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            farm = FarmModel.objects.get(id=farm_id)
        except FarmModel.DoesNotExist:
            return Response({"detail": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)

        if farm.owner != request.user:
            return Response(
                {"detail": "You do not have permission to block users in this farm."},
                status=status.HTTP_403_FORBIDDEN
            )

        members_to_block = User.objects.filter(id__in=user_ids, member_farms=farm)

        if not members_to_block.exists():
            return Response(
                {"detail": "No valid members found to block."},
                status=status.HTTP_400_BAD_REQUEST
            )

        for member in members_to_block:
            farm.members.remove(member)
            farm.blocked.add(member)

        return Response(
            {"detail": f"Blocked {members_to_block.count()} user(s) successfully."},
            status=status.HTTP_200_OK
        )
        
class GetBlockedUsersView(APIView):
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

        blocked_users = farm.blocked.all()
        serializer = MemberSerializer(blocked_users, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
class UnblockUserView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        farm_id = request.data.get("farm")
        user_id = request.data.get("user_id")

        if not farm_id or not user_id:
            return Response({"detail": "Farm ID and User ID are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            farm = FarmModel.objects.get(id=farm_id)
        except FarmModel.DoesNotExist:
            return Response({"detail": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)

        if farm.owner != request.user:
            return Response({"detail": "You do not have permission to unblock users."}, status=status.HTTP_403_FORBIDDEN)

        try:
            user_to_unblock = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"detail": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        if user_to_unblock not in farm.blocked.all():
            return Response({"detail": "User is not blocked."}, status=status.HTTP_400_BAD_REQUEST)

        farm.blocked.remove(user_to_unblock)

        return Response({"detail": "User unblocked successfully."}, status=status.HTTP_200_OK)

class LeaveFarmView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request, id):
        try:
            farm = FarmModel.objects.get(id=id)
        except FarmModel.DoesNotExist:
            return Response({"detail": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)
        
        if farm.owner == request.user:
            return Response({"detail": "Owner cannot leave their own farm. Consider deleting it."},
                            status=status.HTTP_403_FORBIDDEN)

        if request.user not in farm.members.all():
            return Response({"detail": "You are not a member of this farm."}, status=status.HTTP_400_BAD_REQUEST)
        
        farm.members.remove(request.user)
        return Response({"detail": "You have left the farm successfully."}, status=status.HTTP_200_OK)


class DeleteFarmView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def delete(self, request, id):
        try:
            farm = FarmModel.objects.get(id=id)
        except FarmModel.DoesNotExist:
            return Response({"detail": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)
        
        if farm.owner != request.user:
            return Response({"detail": "You do not have permission to delete this farm."}, status=status.HTTP_403_FORBIDDEN)
        
        farm.delete()
        return Response({"detail": "Farm deleted successfully."}, status=status.HTTP_200_OK)
