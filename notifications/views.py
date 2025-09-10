from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import DeviceToken
from .serializers import DeviceTokenSerializer

class RegisterTokenView(APIView):
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
