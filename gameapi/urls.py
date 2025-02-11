# gameapi/urls.py
from django.urls import path
from .views import GamePerformanceUpdateView, GamePredictionView, GameSessionCreateView, RoomPlayersView, CreateRoomPlayerView, CreateRoomView, UpdatePlayerInfoView

urlpatterns = [
    path('session/create/', GameSessionCreateView.as_view(), name='session-create'),
    path('predict/', GamePredictionView.as_view(), name='game-predict'),
    path('room/create/', CreateRoomView.as_view(), name='room-create'),
    path('room/player/create/', CreateRoomPlayerView.as_view(), name='room-player-create'),
    path('room/players/', RoomPlayersView.as_view(), name='room-players'),
    path('room/player/update/<uuid:pk>/', UpdatePlayerInfoView.as_view(), name='room-player-update'),
    path('update', GamePerformanceUpdateView.as_view(), name='game-performance-update'),
]

