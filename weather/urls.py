from django.urls import path
from . import views

urlpatterns = [
    path('forecast/', views.WeatherForeCastView.as_view(), name='weather-forecast'),
]
