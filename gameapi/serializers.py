# gameapi/serializers.py

from rest_framework import serializers
from .models import GamePerformance

class GamePerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = GamePerformance
        fields = [
            'timestamp', 'score', 'lives', 'bubbles_caught', 'bubbles_missed',
            'bubble_count', 'jellyfish_collisions', 'mountain_collisions',
            'hits', 'level', 'reaction_time'
        ]
        read_only_fields = ['timestamp']
