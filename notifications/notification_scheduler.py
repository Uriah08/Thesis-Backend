from apscheduler.schedulers.background import BackgroundScheduler # type: ignore
from .models import DeviceToken
from .helper import send_push_notification

def notify_users():
    print("Running scheduled notification task...")
    tokens = DeviceToken.objects.all()
    for device in tokens:
        send_push_notification(
            device.token,
            "Scheduled Notification",
            "This is a test notification sent by APScheduler"
        )

def start():
    scheduler = BackgroundScheduler()
    scheduler.add_job(notify_users, 'interval', hours=3)
    scheduler.start()
