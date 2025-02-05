# gameapi/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .serializers import GamePerformanceSerializer
from .models import GamePerformance
from .prediction import predict_difficulty_from_model

class GamePerformanceUpdateView(APIView):
    """
    Secure API endpoint to update game performance data.
    """
    def post(self, request, *args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response({"detail": "Authorization header missing or malformed."},
                            status=status.HTTP_401_UNAUTHORIZED)
        token = auth_header.split(' ')[1]
        if token != settings.GAME_UPDATE_TOKEN:
            return Response({"detail": "Invalid token."},
                            status=status.HTTP_403_FORBIDDEN)
        serializer = GamePerformanceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GamePredictionView(APIView):
    """
    Secure API endpoint to predict difficulty based on performance metrics.
    The client sends metrics as query parameters.
    """
    def get(self, request, *args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response({"detail": "Authorization header missing or malformed."},
                            status=status.HTTP_401_UNAUTHORIZED)
        token = auth_header.split(' ')[1]
        if token != settings.GAME_UPDATE_TOKEN:
            return Response({"detail": "Invalid token."},
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