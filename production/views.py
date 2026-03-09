from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from .models import FarmProductionModel
from .serializers import FarmProductionSerializer
from farms.models import FarmModel

class CreateFarmProductionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FarmProductionSerializer(data=request.data, context={"request": request})

        if serializer.is_valid():
            farm_id = request.data.get("farm")
            if not farm_id:
                return Response({"farm": "This field is required."}, status=status.HTTP_400_BAD_REQUEST)

            try:
                farm = FarmModel.objects.get(id=farm_id)
            except FarmModel.DoesNotExist:
                return Response({"farm": "Farm not found."}, status=status.HTTP_400_BAD_REQUEST)

            if farm.owner != request.user:
                return Response({"detail": "Only the farm owner can create production items."},
                                status=status.HTTP_403_FORBIDDEN)

            production = serializer.save(farm=farm)
            response_serializer = FarmProductionSerializer(production, context={"request": request})
            return Response({"detail": "Farm production created successfully.",
                             "farm_production": response_serializer.data},
                            status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ListFarmProductionsView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, farm_id):
        try:
            farm = FarmModel.objects.get(id=farm_id)
        except FarmModel.DoesNotExist:
            return Response({"farm": "Farm not found."}, status=status.HTTP_404_NOT_FOUND)

        # Only owner or members can view
        if not (farm.owner == request.user or request.user in farm.members.all()):
            return Response({"detail": "You are not allowed to view productions for this farm."},
                            status=status.HTTP_403_FORBIDDEN)

        productions = FarmProductionModel.objects.filter(farm=farm)
        serializer = FarmProductionSerializer(productions, many=True, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class RetrieveFarmProductionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, production_id):
        try:
            production = FarmProductionModel.objects.get(id=production_id)
        except FarmProductionModel.DoesNotExist:
            return Response({"detail": "Production not found."}, status=status.HTTP_404_NOT_FOUND)

        # Only owner or members can view
        farm = production.farm
        if not (farm.owner == request.user or request.user in farm.members.all()):
            return Response({"detail": "You are not allowed to view this production."},
                            status=status.HTTP_403_FORBIDDEN)

        serializer = FarmProductionSerializer(production, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class UpdateFarmProductionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def put(self, request, production_id):
        try:
            production = FarmProductionModel.objects.get(id=production_id)
        except FarmProductionModel.DoesNotExist:
            return Response({"detail": "Production not found."}, status=status.HTTP_404_NOT_FOUND)

        # Only owner can update
        if production.farm.owner != request.user:
            return Response({"detail": "Only the farm owner can update this production."},
                            status=status.HTTP_403_FORBIDDEN)

        serializer = FarmProductionSerializer(production, data=request.data, partial=True, context={"request": request})
        if serializer.is_valid():
            serializer.save()
            return Response({"detail": "Production updated successfully.", "farm_production": serializer.data},
                            status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DeleteFarmProductionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def delete(self, request, production_id):
        try:
            production = FarmProductionModel.objects.get(id=production_id)
        except FarmProductionModel.DoesNotExist:
            return Response({"detail": "Production not found."}, status=status.HTTP_404_NOT_FOUND)

        # Only owner can delete
        if production.farm.owner != request.user:
            return Response({"detail": "Only the farm owner can delete this production."},
                            status=status.HTTP_403_FORBIDDEN)

        production.delete()
        return Response({"detail": "Production deleted successfully."}, status=status.HTTP_204_NO_CONTENT)