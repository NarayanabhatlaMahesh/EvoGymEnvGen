from django.db import models
from Experiments.models import Run


class Artifact(models.Model):
    ARTIFACT_TYPES = [
        ("gif", "GIF"),
        ("mp4", "MP4"),
        ("image", "Image"),
        ("checkpoint", "Checkpoint"),
        ("csv", "CSV"),
    ]

    run = models.ForeignKey(Run, on_delete=models.CASCADE)
    artifact_type = models.CharField(max_length=20, choices=ARTIFACT_TYPES)

    path = models.CharField(max_length=500)
    description = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)


class Checkpoint(models.Model):
    run = models.ForeignKey(Run, on_delete=models.CASCADE)
    timestep = models.IntegerField()
    is_best = models.BooleanField(default=False)

    artifact = models.OneToOneField(Artifact, on_delete=models.CASCADE)
