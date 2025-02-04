from django.db import models

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