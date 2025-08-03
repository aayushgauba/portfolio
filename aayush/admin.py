from django.contrib import admin
from .models import Project, Paper, Game, ProfilePhoto, Talk


class ProfilePhotoAdmin(admin.ModelAdmin):
    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        # Delete all other ProfilePhoto objects except the one just saved
        ProfilePhoto.objects.exclude(pk=obj.pk).delete()

admin.site.register(Project)
admin.site.register(Paper)
admin.site.register(Game)
admin.site.register(ProfilePhoto, ProfilePhotoAdmin)
admin.site.register(Talk)
