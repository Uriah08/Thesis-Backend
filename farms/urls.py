from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.CreateFarmView.as_view(), name="create-farm"),
    path('join/', views.JoinFarmView.as_view(), name='join-farm'),
    path('mine/', views.ListUserFarmsView.as_view(), name='mine-farms')
]
