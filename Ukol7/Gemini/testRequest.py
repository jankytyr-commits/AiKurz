import requests

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
api_key = "AIzaSyBuTedKWETrFuT1E541A0iVRH-JOq0cEEw"
headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": api_key
}
data = {
    "contents": [
        {
            "parts": [
                {"text": "Jak funguje umělá inteligence?"}
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())