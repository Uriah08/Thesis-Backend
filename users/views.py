from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import RegisterSerializer, CompleteProfileSerializer
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from notifications.models import DeviceToken

class RegisterView(APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"detail": "User registered successfully."}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class LoginView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response(
                {"detail": "Username and password are required."},
                status=status.HTTP_400_BAD_REQUEST
                )
        
        user = authenticate(username=username, password=password)
        
        if user is not None:
            token, _ = Token.objects.get_or_create(user=user)
            return Response({
                "token": token.key,
                "username": user.username,
                "email": user.email,
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "birthday": user.birthday,
                "address": user.address,
                "is_complete": user.is_complete,
                "profile_picture": user.profile_picture
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                "detail": "Invalid username or password",
            }, status=status.HTTP_401_UNAUTHORIZED)
            
class CompleteProfileView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def put(self, request):
        serializer = CompleteProfileSerializer(
            instance=request.user,
            data=request.data,
            partial=True
        )
        if serializer.is_valid():
            user = serializer.save()
            user.is_complete = True
            user.save()
            print(user)
            return Response({
                "username": user.username,
                "email": user.email,
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "birthday": user.birthday,
                "address": user.address,
                "is_complete": user.is_complete,
                "profile_picture": user.profile_picture
            }, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    
class LogoutView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        token_value = request.data.get("token")
        if not token_value:
            return Response(
                {"error": "Expo push token is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        DeviceToken.objects.filter(
            user=request.user, token=token_value
        ).delete()
            
        try:
            request.user.auth_token.delete()
            return Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)
        except:
            return Response({"detail": "Logout failed."}, status=status.HTTP_400_BAD_REQUEST)


class ChangePasswordView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def put(self, request):
        user = request.user
        old_password = request.data.get("old_password")
        new_password = request.data.get("new_password")
        confirm_password = request.data.get("confirm_password")

        if not old_password or not new_password or not confirm_password:
            return Response(
                {"detail": "All password fields are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not user.check_password(old_password):
            return Response(
                {"detail": "Old password is incorrect."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(new_password) < 8:
            return Response(
                {"detail": "New password must be at least 8 characters long."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if new_password != confirm_password:
            return Response(
                {"detail": "New password and confirm password do not match."},
                status=status.HTTP_400_BAD_REQUEST
            )

        user.set_password(new_password)
        user.save()

        return Response(
            {"detail": "Password changed successfully."},
            status=status.HTTP_200_OK
        )