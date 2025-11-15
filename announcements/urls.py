from django.urls import path
from . import views

urlpatterns = [
    path("create/", views.CreateAnnouncementView.as_view(), name="create-announcement"),
    path("get/<int:id>/", views.GetAnnouncementView.as_view(), name="get-announcement"),
    path("delete/<int:id>/", views.DeleteAnnouncementView.as_view(), name="delete-announcement"),
]