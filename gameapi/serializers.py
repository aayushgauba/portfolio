from rest_framework import serializers
from .models import GamePerformance, Room, RoomPlayer

class RoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Room
        fields = ['id', 'created_at', 'is_active']  
        # Adjust the fields as needed (for example, if you have additional fields such as game_started, etc.)

class RoomPlayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = RoomPlayer
        fields = ['id', 'room', 'player_name', 'joined_at', 'score']
        read_only_fields = ['id', 'joined_at']

class GamePerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = GamePerformance
        fields = [
            'timestamp', 
            'score', 
            'lives', 
            'bubbles_caught', 
            'bubbles_missed',
            'bubble_count', 
            'jellyfish_collisions', 
            'mountain_collisions',
            'hits', 
            'level', 
            'reaction_time'
        ]
        read_only_fields = ['timestamp']
