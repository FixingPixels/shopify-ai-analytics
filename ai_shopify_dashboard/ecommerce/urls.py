from django.urls import path
from .views import get_insights

urlpatterns = [
    path('get-insights/', get_insights),
]
