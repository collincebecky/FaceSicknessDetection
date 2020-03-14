from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from mysite.core import views





urlpatterns = [
    #path('', views.Home.as_view(), name='index'),
    path('', views.run, name='home'),
    
    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
