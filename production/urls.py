from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.CreateFarmProductionView.as_view(), name="create-farm-production"),
    path('list/<int:farm_id>/', views.ListFarmProductionsView.as_view(), name="list-farm-productions"),
    path('retrieve/<int:production_id>/', views.RetrieveFarmProductionView.as_view(), name="retrieve-farm-production"),
    path('update/<int:production_id>/', views.UpdateFarmProductionView.as_view(), name="update-farm-production"),
    path('delete/<int:production_id>/', views.DeleteFarmProductionView.as_view(), name="delete-farm-production"),
]