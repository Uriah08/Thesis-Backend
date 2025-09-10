import requests # type: ignore

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"

def send_push_notification(token, title, body, data=None):
    message = {
        "to": token,
        "sound": "default",
        "title": title,
        "body": body,
        "data": data or {},
    }
    response = requests.post(EXPO_PUSH_URL, json=message)
    return response.json()
