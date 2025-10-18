from django.urls import path
from .views import ScanView

urlpatterns = [
    path('scan/', ScanView.as_view(), name='scan'),
]
