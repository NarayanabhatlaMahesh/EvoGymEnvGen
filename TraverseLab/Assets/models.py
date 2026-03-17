from django.db import models
import hashlib
import uuid


class Morphology(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    body = models.TextField()
    connections = models.TextField()

    voxel_count = models.IntegerField()
    actuator_count = models.IntegerField()

    morphology_hash = models.CharField(max_length=64, unique=True)
    preview_image = models.ImageField(upload_to="morphologies/", blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.morphology_hash:
            h = hashlib.sha256(str(self.body).encode()).hexdigest()
            self.morphology_hash = h
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Morphology {self.morphology_hash[:8]}"


class Environment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    grid_size_x = models.IntegerField()
    grid_size_y = models.IntegerField()

    voxel_grid = models.TextField()
    env_hash = models.CharField(max_length=64, unique=True)

    difficulty_score = models.FloatField(null=True, blank=True)
    preview_image = models.ImageField(upload_to="environments/", blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Env {self.env_hash[:8]}"
