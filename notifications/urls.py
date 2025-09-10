from django.urls import path
from . import views

urlpatterns = [
    path("register-token/", views.RegisterTokenView.as_view(), name="register-token"),
]
