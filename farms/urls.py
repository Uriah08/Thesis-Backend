from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.CreateFarmView.as_view(), name="create-farm"),
    path('join/', views.JoinFarmView.as_view(), name='join-farm'),
    path('mine/', views.ListUserFarmsView.as_view(), name='mine-farms'),
    path("farm/<int:id>/", views.GetFarmView.as_view(), name="get-farm"),
    path("members/<int:id>/", views.GetMembersView.as_view(), name="get-members"),
    path("edit/", views.EditFarmView.as_view(), name="edit-farm"),
    path("change-password/", views.FarmChangePassword.as_view(), name="change-password"),
    path("block-users/", views.BlockUserFarmView.as_view(), name="block-user"),
    path("blocked-users/<int:id>/", views.GetBlockedUsersView.as_view(), name="block-user"),
    path("unblock-user/", views.UnblockUserView.as_view(), name="unblock-user"),
    path("leave/<int:id>/", views.LeaveFarmView.as_view(), name="leave-farm"),
    path("delete/<int:id>/", views.DeleteFarmView.as_view(), name="delete-farm"),
    path("dashboard/<int:farm_id>/", views.GetFarmDashboardView.as_view(), name="farm-dashboard"),
]
