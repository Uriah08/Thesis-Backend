from apscheduler.schedulers.background import BackgroundScheduler
from django.utils import timezone
from announcements.models import AnnouncementModel
from notifications.models import DeviceToken, Notification, Recipient
from .expo import send_push_notification
from django.contrib.auth import get_user_model
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import requests

User = get_user_model()

def delete_expired_announcements():
    print("Running scheduled announcement deletion task...")
    AnnouncementModel.objects.filter(expires_at__lt=timezone.now()).delete()

def notify_users():
    print("Running scheduled notification task...")
    tokens = DeviceToken.objects.all()
    for device in tokens:
        send_push_notification(
            device.token,
            "Scheduled Notification",
            "This is a test notification sent by APScheduler"
        )

def send_daily_weather():
    print("Running tomorrow full-day weather notification...")

    lat = "14.315581"
    lon = "120.742818"
    api_key = "57938c7829f6bab2a497a4c703edace1"
    units = "metric"

    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units={units}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        forecast_list = data.get("list", [])
        city = data.get("city", {}).get("name", "Your Area")

        if not forecast_list:
            print("No forecast data.")
            return

        utc = ZoneInfo("UTC")
        pht = ZoneInfo("Asia/Manila")

        now_pht = datetime.now(pht)
        tomorrow = (now_pht + timedelta(days=1)).date()

        tomorrow_entries = []
        for item in forecast_list:
            dt_utc = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc)
            dt_pht = dt_utc.astimezone(pht)

            if dt_pht.date() == tomorrow:
                tomorrow_entries.append(item)

        if not tomorrow_entries:
            print("No forecast for tomorrow.")
            return

        target_entry = min(
            tomorrow_entries,
            key=lambda x: abs(
                datetime.strptime(x["dt_txt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc)
                .astimezone(pht)
                .hour - 12
            )
        )

        temps = [e["main"]["temp"] for e in tomorrow_entries]
        descriptions = [e["weather"][0]["description"] for e in tomorrow_entries]
        pops = [e.get("pop", 0) for e in tomorrow_entries]

        max_temp = max(temps)
        min_temp = min(temps)

        from collections import Counter
        common_desc = Counter(descriptions).most_common(1)[0][0].title()
        max_pop = max(pops) * 100

        rain = (target_entry.get("pop", 0) * 100)
        wind = target_entry.get("wind", {}).get("speed", 0)
        cloud = target_entry.get("clouds", {}).get("all", 0)

        def rain_desc(r):
            if r < 10: return "no expected rain"
            if r < 30: return f"a slight {r:.0f}% chance of rain"
            if r < 50: return f"a moderate {r:.0f}% chance of rain"
            if r < 80: return f"a high {r:.0f}% chance of rain"
            return f"a very high {r:.0f}% chance of rain"

        def wind_desc(w):
            if w < 10: return "calm winds"
            if w < 15: return "a light breeze"
            if w < 20: return "moderate wind"
            return "strong gusty winds"

        def cloud_desc(c):
            if c < 30: return "mostly clear skies"
            if c < 50: return "partly cloudy skies"
            if c < 70: return "noticeable cloud cover"
            return "overcast skies"

        r_desc = rain_desc(rain)
        w_desc = wind_desc(wind)
        c_desc = cloud_desc(cloud)

        if rain < 10 and wind < 10 and cloud < 50:
            alert = "Excellent"
            alert_msg = (
                f"The weather is perfect for drying fish tomorrow. Expect {c_desc}, "
                f"{w_desc}, and {r_desc}. Ideal for quick and safe drying."
            )
        elif rain < 10 and wind < 15 and cloud < 100:
            alert = "Good"
            alert_msg = (
                f"You can dry fish tomorrow with confidence. Expect {c_desc}, "
                f"{w_desc}, and {r_desc}. Still, have backup cover just in case."
            )
        elif (10 <= rain < 50) or (50 <= cloud < 100) or (15 <= wind < 20):
            alert = "Caution"
            alert_msg = (
                f"Drying fish is possible but not guaranteed tomorrow. {c_desc} and "
                f"{w_desc} may affect drying speed. Also, there's {r_desc}. Stay alert."
            )
        elif (50 <= rain < 80) or wind >= 20 or cloud == 100:
            alert = "Warning"
            alert_msg = (
                f"It's not advisable to dry fish tomorrow. Expect {c_desc}, "
                f"{w_desc}, and {r_desc}. Conditions could compromise drying."
            )
        else:
            alert = "Danger"
            alert_msg = (
                f"Avoid drying fish tomorrow due to {c_desc}, {w_desc}, and {r_desc}. "
                "Conditions are highly unfavorable and risky."
            )

        body = (
            f"{common_desc}. High of {max_temp}°C, low of {min_temp}°C. "
            f"Chance of rain up to {max_pop:.0f}%. "
            f"\n\nFish Drying Alert: {alert}\n{alert_msg}"
        )

        users = User.objects.all()
        notification = Notification.objects.create(
            title=f"{alert} Drying Conditions Expected Tomorrow — {city}",
            body=body,
            type="weather",
            data={
                "city": city,
                "max_temp": max_temp,
                "min_temp": min_temp,
                "description": common_desc,
                "rain_chance": max_pop,
                "drying_alert": alert,
                "drying_message": alert_msg,
                "forecast_day": str(tomorrow)
            }
        )

        Recipient.objects.bulk_create([
            Recipient(notification=notification, user=user)
            for user in users
        ])

        for token in DeviceToken.objects.filter(user__in=users):
            send_push_notification(token.token, notification.title, body)

        print("Tomorrow weather + fish drying rating sent.")

    except Exception as e:
        print("Weather API error:", e)

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(delete_expired_announcements, 'interval', hours=1)
    scheduler.add_job(notify_users, 'interval', hours=3)
    # scheduler.add_job(send_daily_weather, 'interval', seconds=30)
    # scheduler.add_job(
    #     send_daily_weather,
    #     'cron',
    #     hour=7,
    #     minute=0,
    #     timezone='Asia/Manila'
    # )
    scheduler.start()