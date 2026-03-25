"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from django.views.static import serve
from django.conf import settings
import os

def render_nextjs_page(request):
    filepath = os.path.join(settings.BASE_DIR, 'frontend', 'out', 'index.html')
    return serve(request, os.path.basename(filepath), os.path.dirname(filepath))

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # 1. Show the NextJS page on the main URL
    path('', render_nextjs_page, name='home'), 
    
    # 2. Catch-all to serve NextJS CSS, JS, and images (the _next folder)
    re_path(r'^(?P<path>.*)$', serve, {'document_root': os.path.join(settings.BASE_DIR, 'frontend', 'out')}),
]