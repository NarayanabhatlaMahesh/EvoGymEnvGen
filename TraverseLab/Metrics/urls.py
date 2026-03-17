from django.urls import path
from .views import *
# from views import *
urlpatterns = [
    path('metrics/',show_metrics,name='show_metrics'),
]