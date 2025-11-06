from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.TrayCreateView.as_view(), name='tray-create'),
    path('get/<int:session_id>/', views.GetTraysView.as_view(), name='get-trays'),
    path('get/tray/<int:id>/', views.GetTrayView.as_view(), name='get-tray'),
    path('progress/create/', views.CreateTrayProgressView.as_view(), name='tray-progress-create'),
    path("progress/get/<int:tray_id>/", views.GetTrayProgressView.as_view(), name="get_tray_progress"),
    path("harvest/<int:tray_id>/", views.HarvestTrayView.as_view(), name="harvest-tray"),
    path('delete/<int:tray_id>/', views.DeleteTrayView.as_view(), name='delete_tray'),
]
