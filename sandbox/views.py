from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class TestView(APIView):
    def get(self, request):
        return Response({"message": "GET works"}, status=status.HTTP_200_OK)

    def post(self, request):
        return Response({
            "message": "POST works",
            "data": request.data
        }, status=status.HTTP_200_OK)
