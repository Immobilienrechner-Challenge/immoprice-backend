from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('<int:testnum>/', views.test, name='test'),
]