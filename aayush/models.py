import uuid
from django.db import models

class ProfilePhoto(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='profile_photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ProfilePhoto {self.id}"

class Project(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField()
    technologies = models.CharField(max_length=500, help_text="Comma-separated values")  # Store as CSV
    github_link = models.URLField(blank=True, null=True)
    demo_link = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # profile_photo removed as per user request

    def get_technologies_list(self):
        """Return technologies as a list"""
        return [tech.strip() for tech in self.technologies.split(",") if tech.strip()]

    def __str__(self):
        return self.title

class Paper(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    authors = models.TextField(help_text="Comma-separated author names")
    venue = models.CharField(max_length=255)
    year = models.IntegerField()
    abstract = models.TextField()
    tags = models.CharField(max_length=500, help_text="Comma-separated values")  # Store as CSV
    pdf_link = models.URLField(blank=True, null=True)
    publisher_link = models.URLField(blank=True, null=True)
    extra_link = models.URLField(blank=True, null=True)
    # profile_photo removed as per user request

    def get_tags_list(self):
        """Return tags as a list"""
        return [tag.strip() for tag in self.tags.split(",") if tag.strip()]

    def __str__(self):
        return f"{self.title} ({self.year})"

class Game(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField()
    technologies = models.CharField(max_length=500, help_text="Comma-separated values")  # Store as CSV
    github_link = models.URLField(blank=True, null=True)
    demo_link = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def get_technologies_list(self):
        """Return technologies as a list"""
        return [tech.strip() for tech in self.technologies.split(",") if tech.strip()]

    def __str__(self):
        return self.title