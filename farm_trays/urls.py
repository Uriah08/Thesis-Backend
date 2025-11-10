from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.CreateFarmTrayView.as_view(), name="create-farm-tray"),
    path('get/<int:farm_id>/', views.GetFarmTraysView.as_view(), name="get-farm-trays"),
    path('get/tray/<int:tray_id>/', views.GetFarmTrayByIdView.as_view(), name="get-farm-tray"),
    path('maintenance/<int:tray_id>/', views.TrayMaintenanceView.as_view(), name="tray-maintenance"),
    path('rename/<int:tray_id>/', views.RenameTrayView.as_view(), name="rename-tray"),
    path('delete/<int:tray_id>/', views.DeleteTrayView.as_view(), name="delete-tray"),
]
