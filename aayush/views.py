from django.shortcuts import render
from .models import Project, Paper, Game

def portfolio_view(request):
    projects = Project.objects.all().order_by('-created_at')  # Latest projects first
    papers = Paper.objects.all().order_by('-year')  # Latest papers first
    games = Game.objects.all().order_by('-created_at')  # Latest games first

    context = {
        'projects': projects,
        'papers': papers,
        'games': games,
    }
    
    return render(request, 'aayush/aayush.html', context)