from django.urls import path
from .views import get_insights


urlpatterns = [    
    path('get_insights/', get_insights,name='get_insights')
]