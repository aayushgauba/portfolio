# gameapi/urls.py

from django.urls import path
from .views import GamePerformanceUpdateView, GamePredictionView

urlpatterns = [
    path('update/', GamePerformanceUpdateView.as_view(), name='game-update'),
    path('predict/', GamePredictionView.as_view(), name='game-predict'),
]
