# Generated by Django 5.1.5 on 2025-02-24 20:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gameapi', '0004_room_game_started'),
    ]

    operations = [
        migrations.AddField(
            model_name='room',
            name='leader_id',
            field=models.PositiveIntegerField(blank=True, help_text='Primary key of the RoomPlayer who is the leader.', null=True),
        ),
    ]
