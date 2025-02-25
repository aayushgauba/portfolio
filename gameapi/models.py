from django.db import models
import secrets
import random
import uuid

def generate_room_id():
    """
    Generates a random 10-digit number as a string.
    For example: "1234567890"
    """
    return str(random.randint(10**9, 10**10 - 1))

class Room(models.Model):
    """
    Represents a multiplayer game room.
    The primary key is a unique random 10-digit number (as a string).
    """
    id = models.CharField(
        max_length=10,
        primary_key=True,
        default=generate_room_id,  # Use the named function instead of a lambda
        unique=True,
        editable=False,
        help_text="Unique 10-digit room ID"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(
        default=True,
        help_text="Indicates whether the room is active"
    )
    game_started = models.BooleanField(default=False)
    leader_id = models.PositiveIntegerField(
        blank=True, null=True,
        help_text="Primary key of the RoomPlayer who is the leader."
    )
    def __str__(self):
        return f"Room {self.id}"

class RoomPlayer(models.Model):
    """
    Represents a player assigned to a room.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    room = models.ForeignKey(
        Room,
        on_delete=models.CASCADE,
        related_name='players',
        help_text="The room the player is assigned to"
    )
    player_name = models.CharField(
        max_length=255,
        help_text="Player's display name"
    )
    joined_at = models.DateTimeField(auto_now_add=True)
    score = models.IntegerField(default=0, help_text="Player score")

    def __str__(self):
        return f"{self.player_name} in Room {self.room.id}"

class GamePerformance(models.Model):
    """
    Records a player's score for a game session in a given room.
    """
    room = models.ForeignKey(
        Room,
        on_delete=models.CASCADE,
        related_name='performances',
        help_text="The room associated with this performance"
    )
    player = models.ForeignKey(
        RoomPlayer,
        on_delete=models.CASCADE,
        related_name='performances',
        help_text="The player whose score is recorded"
    )
    score = models.IntegerField(default=0, help_text="Player's score in the session")
    created_at = models.DateTimeField(auto_now_add=True)
    leader_id = models.PositiveIntegerField(
            blank=True, null=True,
            help_text="Primary key of the RoomPlayer who is the leader."
        )
    def __str__(self):
        return f"{self.player} scored {self.score} in Room {self.room.id} at {self.created_at}"
    
class GamePerformance(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    score = models.IntegerField(default=0)
    lives = models.IntegerField(default=3)
    bubbles_caught = models.IntegerField(default=0)
    bubbles_missed = models.IntegerField(default=0)
    bubble_count = models.IntegerField(default=0)
    jellyfish_collisions = models.IntegerField(default=0)
    mountain_collisions = models.IntegerField(default=0)
    hits = models.IntegerField(default=0)
    level = models.IntegerField(default=1)
    reaction_time = models.FloatField(null=True, blank=True)
    def __str__(self):
        return f"{self.timestamp} | Score: {self.score}, Level: {self.level}"

class GameSession(models.Model):
    """
    Represents a unique game session with its own update token.
    """
    created = models.DateTimeField(auto_now_add=True)
    token = models.CharField(max_length=100, blank=True, unique=True)

    def save(self, *args, **kwargs):
        if not self.token:
            self.token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Session {self.pk} at {self.created}"