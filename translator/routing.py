from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # NextJS will connect to ws://localhost:8000/ws/translate/
    re_path(r'ws/translate/$', consumers.SignLanguageConsumer.as_asgi()),
]