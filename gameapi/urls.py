# gameapi/urls.py
from django.urls import path
from .views import GamePerformanceUpdateView, GamePredictionView, GameSessionCreateView

urlpatterns = [
    path('update/', GamePerformanceUpdateView.as_view(), name='game-update'),
    path('predict/', GamePredictionView.as_view(), name='game-predict'),
    path('session/create/', GameSessionCreateView.as_view(), name='game-session-create'),
]

