from django.shortcuts import render
from .models import Project, Paper, Game, Talk

def portfolio_view(request):
    projects = Project.objects.all().order_by('-created_at')  # Latest projects first
    papers = Paper.objects.all().order_by('-year')  # Latest papers first
    games = Game.objects.all().order_by('-created_at')  # Latest games first

    from .models import ProfilePhoto
    profile_photo = ProfilePhoto.objects.first()
    talks = Talk.objects.all().order_by('-date')
    context = {
        'projects': projects,
        'papers': papers,
        'games': games,
        'profile_photo': profile_photo,
        'talks': talks,
    }
    return render(request, 'aayush/aayush.html', context)