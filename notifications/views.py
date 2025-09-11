from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import DeviceToken, Recipient
from .serializers import DeviceTokenSerializer, NotificationSerializer
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import get_user_model
from .helper import send_push_notification

User = get_user_model()

class RegisterTokenView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        serializer = DeviceTokenSerializer(data=request.data)

        if serializer.is_valid():
            token = serializer.validated_data["token"]
            user = request.user

            if DeviceToken.objects.filter(user=user, token=token).exists():
                return Response({"message": None}, status=status.HTTP_200_OK)

            DeviceToken.objects.create(user=user, token=token)
            return Response({"message": "Token saved"}, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class NotificationCreateView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = NotificationSerializer(data=request.data)
        if serializer.is_valid():
            notification = serializer.save()
            
            user_ids = request.data.get("user_ids", [])
            if user_ids:
                users = User.objects.filter(id__in=user_ids)
            else:
                users = User.objects.all()
            
            recipients = [Recipient(notification=notification, user=user) for user in users]
            Recipient.objects.bulk_create(recipients)
            
            if user_ids:
                device_tokens = DeviceToken.objects.filter(user__in=users)
            else:
                device_tokens = DeviceToken.objects.all()

            for token in device_tokens:
                send_push_notification(token.token, notification.title, notification.body)
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
