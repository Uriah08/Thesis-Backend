from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class WeatherForeCastView(APIView):
    def get(self, request):
        lat = request.query_params.get('lat', '14.315581')
        lon = request.query_params.get('lon', '120.742818')
        api_key = '57938c7829f6bab2a497a4c703edace1'
        units = request.query_params.get('units', 'metric')

        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units={units}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            city_info = {
                "country": data.get("city", {}).get("country"),
                "name": data.get("city", {}).get("name")
            }

            forecast_list = data.get("list", [])
            if not forecast_list:
                return Response({"error": "No forecast data available."}, status=status.HTTP_404_NOT_FOUND)

            # Timezones
            utc_tz = ZoneInfo("UTC")
            ph_tz = ZoneInfo("Asia/Manila")

            # First item (converted to PHT)
            first = forecast_list[0]
            first_utc = datetime.strptime(first["dt_txt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc_tz)
            first_ph = first_utc.astimezone(ph_tz)
            cutoff_dt = first_ph + timedelta(days=2)

            first_item = {
                "datetime": first_ph.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": first["main"]["temp"],
                "description": first["weather"][0]["description"],
                "icon": first["weather"][0]["icon"],
                "clouds": first["clouds"]["all"],
                "wind_speed": first["wind"]["speed"],
                "pop": first.get("pop", 0)
            }

            future_forecast = []
            for item in forecast_list[1:]:
                item_utc = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc_tz)
                item_ph = item_utc.astimezone(ph_tz)
                if item_ph <= cutoff_dt:
                    future_forecast.append({
                        "datetime": item_ph.strftime("%Y-%m-%d %H:%M:%S"),
                        "temperature": item["main"]["temp"],
                        "description": item["weather"][0]["description"],
                        "icon": item["weather"][0]["icon"],
                        "clouds": item["clouds"]["all"],
                        "wind_speed": item["wind"]["speed"],
                        "pop": item.get("pop", 0)
                    })

            return Response({
                "city": city_info,
                "first_item": first_item,
                "future_forecast": future_forecast
            }, status=status.HTTP_200_OK)

        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
