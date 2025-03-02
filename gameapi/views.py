from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .serializers import (
    GamePerformanceSerializer,
    RoomSerializer,
    RoomPlayerSerializer
)
from .models import GamePerformance, GameSession, Room, RoomPlayer
from .prediction import predict_difficulty_from_model
import random

def validate_session(request):
    """
    Validates that a valid session_id and token are provided in the request.
    Returns a tuple: (session, error_response) where error_response is None if valid.
    """
    session_id = request.data.get("session_id") or request.query_params.get("session_id")
    if not session_id:
        return None, Response({"detail": "Missing session_id."}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        session = GameSession.objects.get(pk=session_id)
    except GameSession.DoesNotExist:
        return None, Response({"detail": "Invalid session_id."}, status=status.HTTP_400_BAD_REQUEST)
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, Response({"detail": "Authorization header missing or malformed."},
                              status=status.HTTP_401_UNAUTHORIZED)
    token = auth_header.split(" ")[1]
    if token != session.token:
        return None, Response({"detail": "Invalid token for this session."},
                              status=status.HTTP_403_FORBIDDEN)
    return session, None

class RoomStartView(APIView):
    """
    API endpoint for the room leader to start the game.
    Only the room leader (the first RoomPlayer who joined the room) may start the game.
    
    Expected payload:
      {
          "session_id": "<session_id>",
          "room": "<room_id>",
          "player_id": "<player_id>"
      }
    """
    def post(self, request, *args, **kwargs):
        session, error_response = validate_session(request)
        if error_response:
            return error_response
        
        room_id = request.data.get("room")
        if not room_id:
            return Response({"detail": "Missing room parameter."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            room = Room.objects.get(pk=room_id)
        except Room.DoesNotExist:
            return Response({"detail": "Room not found."}, status=status.HTTP_404_NOT_FOUND)

        player_id = request.data.get("player_id")
        if not player_id:
            return Response({"detail": "Missing player_id."}, status=status.HTTP_400_BAD_REQUEST)

        # Check if the caller is the official leader
        if str(room.leader_id) != str(player_id):
            return Response({"detail": "Only the designated leader can start the game."},
                            status=status.HTTP_403_FORBIDDEN)

        # Mark the room as started
        room.game_started = True
        room.save()
        return Response({"detail": "Game started."}, status=status.HTTP_200_OK)

class GameSessionCreateView(APIView):
    def post(self, request, *args, **kwargs):
        session = GameSession.objects.create()
        return Response({"session_id": session.pk, "token": session.token}, status=status.HTTP_201_CREATED)

class RoomPlayersView(APIView):
    def get(self, request, *args, **kwargs):
        session, error_response = validate_session(request)
        if error_response:
            return error_response
        room_id = request.query_params.get("room")
        if not room_id:
            return Response({"detail": "Missing room parameter."}, status=400)
        try:
            room = Room.objects.get(pk=room_id)
        except Room.DoesNotExist:
            return Response({"detail": "Room not found."}, status=404)
        players = RoomPlayer.objects.filter(room=room)
        players_data = RoomPlayerSerializer(players, many=True).data
        data = {
            "id": room.pk,
            "leader_id": room.leader_id,       
            "game_started": room.game_started,
            "players": players_data
        }
        return Response(data, status=200)

class CreateRoomView(APIView):
    """
    Creates a new Room.
    Requires a valid session (via session_id and token).
    The Room’s primary key is set to a random 10-digit number.
    """
    def post(self, request, *args, **kwargs):
        session, error_response = validate_session(request)
        if error_response:
            return error_response

        # Create a 10-digit random room ID.
        room_id = random.randint(10**9, 10**10 - 1)
        room = Room.objects.create(id=room_id)
        serializer = RoomSerializer(room)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

class CreateRoomPlayerView(APIView):
    """
    Creates (joins) a player into a room.
    Expected payload:
      {
          "session_id": "<session_id>",
          "room": "<room_id>",
          "player_name": "<display name>"
      }
    """
    def post(self, request, *args, **kwargs):
        session, error_response = validate_session(request)
        if error_response:
            return error_response
        
        serializer = RoomPlayerSerializer(data=request.data)
        if serializer.is_valid():
            player = serializer.save()
            
            # Now check if the room has no leader yet.
            room_instance = player.room
            if room_instance.leader_id is None:
                room_instance.leader_id = player.id
                room_instance.save()
            
            return Response(RoomPlayerSerializer(player).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RoomStartView(APIView):
    """
    API endpoint for the room leader to start the game.
    Only the room leader (the first RoomPlayer created in the room) may start the game.
    Expected payload:
      {
          "session_id": "<session_id>",
          "room": "<room_id>",
          "player_id": "<player_id>"
      }
    """
    def post(self, request, *args, **kwargs):
        # Validate the session and token.
        session, error_response = validate_session(request)
        if error_response:
            return error_response
        
        room_id = request.data.get("room")
        if not room_id:
            return Response({"detail": "Missing room parameter."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            room = Room.objects.get(pk=room_id)
        except Room.DoesNotExist:
            return Response({"detail": "Room not found."}, status=status.HTTP_404_NOT_FOUND)
        
        # Retrieve all players in the room ordered by join time.
        players = RoomPlayer.objects.filter(room=room).order_by("joined_at")
        if not players.exists():
            return Response({"detail": "No players in room."}, status=status.HTTP_400_BAD_REQUEST)
        leader = players.first()
        
        player_id = request.data.get("player_id")
        if not player_id:
            return Response({"detail": "Missing player_id."}, status=status.HTTP_400_BAD_REQUEST)
        
        if str(leader.pk) != str(player_id):
            return Response({"detail": "Only the room leader can start the game."},
                            status=status.HTTP_403_FORBIDDEN)
        
        # Mark the room as started.
        room.game_started = True
        room.save()
        return Response({"detail": "Game started."}, status=status.HTTP_200_OK)

class GamePredictionView(APIView):
    """
    Secure API endpoint to predict difficulty based on performance metrics.
    The client sends metrics as query parameters along with a session_id.
    """
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get("session_id")
        if not session_id:
            return Response({"detail": "Missing session_id."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            session = GameSession.objects.get(pk=session_id)
        except GameSession.DoesNotExist:
            return Response({"detail": "Invalid session_id."}, status=status.HTTP_400_BAD_REQUEST)
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response({"detail": "Authorization header missing or malformed."},
                            status=status.HTTP_401_UNAUTHORIZED)
        token = auth_header.split(' ')[1]
        if token != session.token:
            return Response({"detail": "Invalid token for this session."},
                            status=status.HTTP_403_FORBIDDEN)
        try:
            bubbles_caught = int(request.query_params.get("bubbles_caught", 0))
            bubbles_missed = int(request.query_params.get("bubbles_missed", 0))
            jellyfish_collisions = int(request.query_params.get("jellyfish_collisions", 0))
            mountain_collisions = int(request.query_params.get("mountain_collisions", 0))
        except ValueError:
            return Response({"detail": "Invalid query parameter format."},
                            status=status.HTTP_400_BAD_REQUEST)
        performance_data = {
            "bubbles_caught": bubbles_caught,
            "bubbles_missed": bubbles_missed,
            "jellyfish_collisions": jellyfish_collisions,
            "mountain_collisions": mountain_collisions,
        }
        predicted_difficulty = predict_difficulty_from_model(performance_data)
        return Response({"predicted_difficulty": predicted_difficulty}, status=status.HTTP_200_OK)

class UpdatePlayerInfoView(APIView):
    """
    Updates a player's information (e.g. score).
    The endpoint expects a session_id and token along with the update data.
    """
    def put(self, request, pk, *args, **kwargs):
        session, error_response = validate_session(request)
        if error_response:
            return error_response
        try:
            player = RoomPlayer.objects.get(pk=pk)
        except RoomPlayer.DoesNotExist:
            return Response({"detail": "Player not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = RoomPlayerSerializer(player, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class GamePerformanceUpdateView(APIView):
    """
    Secure API endpoint to update game performance data.
    The client must provide a valid session_id and token.
    """
    def post(self, request, *args, **kwargs):
        session_id = request.data.get("session_id") or request.query_params.get("session_id")
        if not session_id:
            return Response({"detail": "Missing session_id."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            session = GameSession.objects.get(pk=session_id)
        except GameSession.DoesNotExist:
            return Response({"detail": "Invalid session_id."}, status=status.HTTP_400_BAD_REQUEST)
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response({"detail": "Authorization header missing or malformed."},
                            status=status.HTTP_401_UNAUTHORIZED)
        token = auth_header.split(' ')[1]
        if token != session.token:
            return Response({"detail": "Invalid token for this session."},
                            status=status.HTTP_403_FORBIDDEN)
        serializer = GamePerformanceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

