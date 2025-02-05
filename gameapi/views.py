# gameapi/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .serializers import GamePerformanceSerializer
from .models import GamePerformance, GameSession
from .prediction import predict_difficulty_from_model

class GameSessionCreateView(APIView):
    """
    API endpoint to create a new game session.
    Returns the session ID and unique token.
    """
    def post(self, request, *args, **kwargs):
        session = GameSession.objects.create()
        return Response({"session_id": session.pk, "token": session.token}, status=status.HTTP_201_CREATED)

# gameapi/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .serializers import GamePerformanceSerializer
from .models import GamePerformance, GameSession  # Make sure GameSession is defined in your models
from .prediction import predict_difficulty_from_model

class GamePredictionView(APIView):
    """
    Secure API endpoint to predict difficulty based on performance metrics.
    The client sends metrics as query parameters along with a session_id.
    The session-specific token (in the Authorization header) is validated against the GameSession.
    """
    def get(self, request, *args, **kwargs):
        # Retrieve session_id from query parameters
        session_id = request.query_params.get("session_id")
        if not session_id:
            return Response({"detail": "Missing session_id."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the game session
        try:
            session = GameSession.objects.get(pk=session_id)
        except GameSession.DoesNotExist:
            return Response({"detail": "Invalid session_id."}, status=status.HTTP_400_BAD_REQUEST)

        # Validate the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response({"detail": "Authorization header missing or malformed."},
                            status=status.HTTP_401_UNAUTHORIZED)
        token = auth_header.split(' ')[1]
        if token != session.token:
            return Response({"detail": "Invalid token for this session."},
                            status=status.HTTP_403_FORBIDDEN)

        # Get performance metrics from query parameters
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


class GamePerformanceUpdateView(APIView):
    """
    Secure API endpoint to update game performance data.
    The client must provide a valid session_id and token.
    """
    def post(self, request, *args, **kwargs):
        # Retrieve session_id from request data or query parameters
        session_id = request.data.get("session_id") or request.query_params.get("session_id")
        if not session_id:
            return Response({"detail": "Missing session_id."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the game session
        try:
            session = GameSession.objects.get(pk=session_id)
        except GameSession.DoesNotExist:
            return Response({"detail": "Invalid session_id."}, status=status.HTTP_400_BAD_REQUEST)

        # Validate the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response({"detail": "Authorization header missing or malformed."},
                            status=status.HTTP_401_UNAUTHORIZED)
        token = auth_header.split(' ')[1]
        if token != session.token:
            return Response({"detail": "Invalid token for this session."},
                            status=status.HTTP_403_FORBIDDEN)

        # Proceed with updating game performance
        serializer = GamePerformanceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
