from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.CreateFarmSession.as_view(), name="create-farm-session"),
]
