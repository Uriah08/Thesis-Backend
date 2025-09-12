from django.urls import path
from . import views

urlpatterns = [
    path("register-token/", views.RegisterTokenView.as_view(), name="register-token"),
    path("create-notification/", views.NotificationCreateView.as_view(), name="create-notification"),
    path("my-notifications/", views.MyNotificationsView.as_view(), name="my-notifications"),
    path("my-notification", views.MyNotificationView.as_view(), name="my-notification"),
    path("read-notifications/", views.ReadNotificationView.as_view(), name="read-notifications"),
    path("delete-notifications/", views.DeleteNotificationsView.as_view(), name="delete-notifications")
]
