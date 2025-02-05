from django.db import models
import secrets

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