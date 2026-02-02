from apscheduler.schedulers.background import BackgroundScheduler
from django.utils import timezone
from announcements.models import AnnouncementModel
from notifications.models import DeviceToken, Notification, Recipient
from .expo import send_push_notification
from django.contrib.auth import get_user_model
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import requests
from collections import Counter

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
    print("Running weather notification...")

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
        city = data.get("city", {}).get("name", "Naic")

        if not forecast_list:
            return

        utc = ZoneInfo("UTC")
        pht = ZoneInfo("Asia/Manila")
        now_pht = datetime.now(pht)

        # Tomorrow + next day
        target_days = [
            now_pht.date() + timedelta(days=1),
            now_pht.date() + timedelta(days=2),
        ]

        full_body = ""
        highest_alert = "Excellent"
        alert_rank = {
            "Excellent": 0,
            "Good": 1,
            "Caution": 2,
            "Warning": 3,
            "Danger": 4,
        }

        for target_date in target_days:
            day_entries = []

            for item in forecast_list:
                dt_utc = datetime.strptime(
                    item["dt_txt"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=utc)
                dt_pht = dt_utc.astimezone(pht)

                if dt_pht.date() == target_date:
                    day_entries.append(item)

            if not day_entries:
                continue

            # Closest to 12 PM
            target_entry = min(
                day_entries,
                key=lambda x: abs(
                    datetime.strptime(x["dt_txt"], "%Y-%m-%d %H:%M:%S")
                    .replace(tzinfo=utc)
                    .astimezone(pht)
                    .hour - 12
                )
            )

            temps = [e["main"]["temp"] for e in day_entries]
            descriptions = [e["weather"][0]["description"] for e in day_entries]

            max_temp = max(temps)
            min_temp = min(temps)
            common_desc = Counter(descriptions).most_common(1)[0][0].title()

            rain_percent = target_entry.get("pop", 0) * 100
            cloud = target_entry.get("clouds", {}).get("all", 0)

            def rain_desc(r):
                if r == 0: return "no expected rain"
                if r < 30: return f"a slight {r:.0f}% chance of rain"
                if r < 60: return f"a moderate {r:.0f}% chance of rain"
                if r < 80: return f"a high {r:.0f}% chance of rain"
                return f"a very high {r:.0f}% chance of rain"

            def cloud_desc(c):
                if c < 30: return "mostly clear skies"
                if c < 60: return "partly cloudy skies"
                if c < 85: return "noticeable cloud cover"
                return "overcast skies"

            rain_text = rain_desc(rain_percent)
            cloud_text = cloud_desc(cloud)

            if rain_percent == 0 and cloud < 50:
                alert = "Excellent"
                message = f"Ideal conditions for drying fish: {cloud_text}, and {rain_text}."
            elif rain_percent == 0:
                alert = "Good"
                message = f"Good weather for drying fish with {cloud_text}, and {rain_text}."
            elif rain_percent <= 80:
                alert = "Caution"
                message = f"Be cautious: {cloud_text}, and {rain_text}. Drying may be slow or risky."
            elif rain_percent < 99:
                alert = "Warning"
                message = f"Drying fish is not recommended due to {cloud_text}, and {rain_text}."
            else:
                alert = "Danger"
                message = f"Avoid drying fish. Extreme conditions: {cloud_text}, and {rain_text}."

            if alert_rank[alert] > alert_rank[highest_alert]:
                highest_alert = alert

            date_str = target_date.strftime("%B %d, %Y")

            full_body += (
                f"{date_str}\n"
                f"{common_desc}. High {max_temp}°C / Low {min_temp}°C\n"
                f"Chance of rain up to {rain_percent:.0f}%\n"
                f"Fish Drying Alert: {alert}\n{message}\n\n"
            )

        if not full_body:
            return

        # ---- TITLE DATE RANGE ----
        start_date, end_date = target_days

        if start_date.month == end_date.month:
            title_date = f"{start_date.strftime('%B %d')}–{end_date.strftime('%d, %Y')}"
        else:
            title_date = f"{start_date.strftime('%B %d, %Y')} – {end_date.strftime('%B %d, %Y')}"

        users = User.objects.all()

        notification = Notification.objects.create(
            title=f"Drying Conditions Expected Tomorrow — {city}, {title_date}",
            body=full_body.strip(),
            type="weather",
            data={
                "city": city,
                "dates": [d.strftime("%Y-%m-%d") for d in target_days],
                "alert": highest_alert,
            }
        )

        Recipient.objects.bulk_create([
            Recipient(notification=notification, user=user)
            for user in users
        ])

        for token in DeviceToken.objects.filter(user__in=users):
            send_push_notification(token.token, notification.title, notification.body)

        print("Weather notification sent successfully.")

    except Exception as e:
        print("Weather API error:", e)
        

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(delete_expired_announcements, 'interval', hours=1)
    scheduler.add_job(notify_users, 'interval', hours=3)
    
    # scheduler.add_job(send_daily_weather, 'interval', seconds=10)
    
    # scheduler.add_job(
    #     send_daily_weather,
    #     'cron',
    #     hour=13,
    #     minute=41,
    #     timezone='Asia/Manila'
    # )

    scheduler.start()
    
    
    # def send_daily_weather():
#     print("Running tomorrow full-day weather notification...")

#     lat = "14.315581"
#     lon = "120.742818"
#     api_key = "57938c7829f6bab2a497a4c703edace1"
#     units = "metric"

#     url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units={units}"

#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()

#         forecast_list = data.get("list", [])
#         city = data.get("city", {}).get("name", "Your Area")

#         if not forecast_list:
#             print("No forecast data.")
#             return

#         utc = ZoneInfo("UTC")
#         pht = ZoneInfo("Asia/Manila")

#         now_pht = datetime.now(pht)
#         tomorrow = (now_pht + timedelta(days=1)).date()

#         tomorrow_entries = []
#         for item in forecast_list:
#             dt_utc = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc)
#             dt_pht = dt_utc.astimezone(pht)

#             if dt_pht.date() == tomorrow:
#                 tomorrow_entries.append(item)

#         if not tomorrow_entries:
#             print("No forecast for tomorrow.")
#             return

#         # Pick entry closest to 12PM
#         target_entry = min(
#             tomorrow_entries,
#             key=lambda x: abs(
#                 datetime.strptime(x["dt_txt"], "%Y-%m-%d %H:%M:%S")
#                 .replace(tzinfo=utc)
#                 .astimezone(pht)
#                 .hour - 12
#             )
#         )

#         temps = [e["main"]["temp"] for e in tomorrow_entries]
#         descriptions = [e["weather"][0]["description"] for e in tomorrow_entries]
#         pops = [e.get("pop", 0) for e in tomorrow_entries]

#         max_temp = max(temps)
#         min_temp = min(temps)

#         from collections import Counter
#         common_desc = Counter(descriptions).most_common(1)[0][0].title()

#         rainPercent = target_entry.get("pop", 0) * 100
#         cloud = target_entry.get("clouds", {}).get("all", 0)

#         def getRainDescription(r):
#             if r == 0: return "no expected rain"
#             if r < 30: return f"a slight {r:.0f}% chance of rain"
#             if r < 60: return f"a moderate {r:.0f}% chance of rain"
#             if r < 80: return f"a high {r:.0f}% chance of rain"
#             return f"a very high {r:.0f}% chance of rain"

#         def getCloudDescription(c):
#             if c < 30: return "mostly clear skies"
#             if c < 60: return "partly cloudy skies"
#             if c < 85: return "noticeable cloud cover"
#             return "overcast skies"

#         rainDesc = getRainDescription(rainPercent)
#         cloudDesc = getCloudDescription(cloud)

#         if rainPercent == 0 and cloud < 50:
#             alert = "Excellent"
#             message = f"Ideal conditions for drying fish: {cloudDesc}, and {rainDesc}."

#         elif rainPercent == 0 and cloud <= 100:
#             alert = "Good"
#             message = f"Good weather for drying fish with {cloudDesc}, and {rainDesc}."

#         elif rainPercent <= 80 and rainPercent > 0 and cloud <= 100:
#             alert = "Caution"
#             message = (
#                 f"Be cautious: {cloudDesc}, and {rainDesc}. "
#                 "Drying may be slow or risky."
#             )

#         elif rainPercent > 80 and rainPercent < 99:
#             alert = "Warning"
#             message = (
#                 f"Drying fish is not recommended due to {cloudDesc}, and {rainDesc}."
#             )

#         else:
#             alert = "Danger"
#             message = (
#                 f"Avoid drying fish. Extreme conditions: {cloudDesc}, and {rainDesc}."
#             )

#         # ----------------------------
#         # BUILD NOTIFICATION BODY
#         # ----------------------------

#         body = (
#             f"{common_desc}. High of {max_temp}°C, low of {min_temp}°C.\n"
#             f"Chance of rain up to {rainPercent:.0f}%.\n\n"
#             f"Fish Drying Alert: {alert}\n{message}"
#         )

#         users = User.objects.all()

#         notification = Notification.objects.create(
#             title=f"{alert} Drying Conditions Expected Tomorrow — {city}",
#             body=body,
#             type="weather",
#             data={
#                 "city": city,
#                 "max_temp": max_temp,
#                 "min_temp": min_temp,
#                 "description": common_desc,
#                 "rain_chance": rainPercent,
#                 "drying_alert": alert,
#                 "drying_message": message,
#                 "forecast_day": str(tomorrow)
#             }
#         )

#         Recipient.objects.bulk_create([
#             Recipient(notification=notification, user=user)
#             for user in users
#         ])

#         for token in DeviceToken.objects.filter(user__in=users):
#             send_push_notification(token.token, notification.title, body)

#         print("Tomorrow weather + fish drying rating sent.")

#     except Exception as e:
#         print("Weather API error:", e)
