# """
# ASGI config for backend project.

# It exposes the ASGI callable as a module-level variable named ``application``.

# For more information on this file, see
# https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
# """

# import os

# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# application = get_asgi_application()



import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import translator.routing # Import the routes we just made!

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# This is our traffic cop!
application = ProtocolTypeRouter({
    # 1. Standard HTTP traffic (normal web requests) goes here:
    "http": get_asgi_application(),
    
    # 2. Live WebSocket traffic (NextJS video frames) will go here:
    # "websocket": URLRouter(...) <-- We will build this part next!
    # Traffic Cop: Send all websocket traffic to our translator routing file
    "websocket": AuthMiddlewareStack(
        URLRouter(
            translator.routing.websocket_urlpatterns
        )
    ),
})