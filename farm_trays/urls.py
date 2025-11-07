from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.CreateFarmTrayView.as_view(), name="create-farm-tray"),
    path('get/<int:farm_id>/', views.GetFarmTraysView.as_view(), name="get-farm-trays"),
    path('get/tray/<int:tray_id>/', views.GetFarmTrayByIdView.as_view(), name="get-farm-tray"),
]
