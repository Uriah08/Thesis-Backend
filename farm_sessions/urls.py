from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.CreateFarmSessionView.as_view(), name="create-farm-session"),
    path('get/<int:farm_id>/', views.GetFarmSessionsView.as_view(), name="get-farm-sessions"),
]
