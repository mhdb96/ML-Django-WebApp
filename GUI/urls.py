from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='gui-home'),
    path('about/', views.about, name='gui-about'),
    path('evaluate', views.evaluate, name='gui-evaluate'),
]
