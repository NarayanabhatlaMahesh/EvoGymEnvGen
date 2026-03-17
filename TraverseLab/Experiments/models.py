from django.db import models
import uuid


class Experiment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class TimestampEnvGenerated(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    timestamp = models.DateTimeField()

class Curriculum(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(TimestampEnvGenerated, on_delete=models.CASCADE, related_name="curriculums")
    difficulty = models.CharField(max_length=20)
    object_size = models.IntegerField(blank=True)
    object_count = models.IntegerField(default=0,blank=True)
    description = models.TextField(blank=True)

class VisualiseEnvs(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    path = models.CharField(max_length=2000)
    generation = models.IntegerField()
    curriculum = models.ForeignKey(Curriculum, on_delete=models.CASCADE, related_name="visualise_envs")

class EnvImages(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    visualise_env = models.ForeignKey(VisualiseEnvs, on_delete=models.CASCADE, related_name="images")
    image_path = models.CharField(max_length=2000)

class Run(models.Model):
    STATUS_CHOICES = [
        ("running", "Running"),
        ("finished", "Finished"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, related_name="runs"
    )

    mlflow_run_id = models.CharField(max_length=64, blank=True, null=True)
    parent_run = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL
    )

    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    seed = models.IntegerField()
    git_commit_hash = models.CharField(max_length=40, blank=True)

    total_timesteps = models.IntegerField()
    total_updates = models.IntegerField()

    start_time = models.DateTimeField(null=True)
    end_time = models.DateTimeField(null=True)

    notes = models.TextField(blank=True)

    def __str__(self):
        return f"Run {self.id}"


class RunConfig(models.Model):
    run = models.OneToOneField(Run, on_delete=models.CASCADE)

    ppo_params = models.TextField()
    reward_params = models.TextField()
    env_params = models.TextField()
    mutation_params = models.TextField(blank=True, null=True)
    noise_params = models.TextField(blank=True, null=True)

class GeneratedEnv(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    visualise_Envs = models.ForeignKey(VisualiseEnvs, on_delete=models.CASCADE, related_name="GeneratedEnv")
    avgIOU = models.IntegerField()
    avgNCD = models.IntegerField()
    avgPPO = models.IntegerField()
    maxIOU = models.IntegerField()
    maxNCD = models.IntegerField()
    maxPPO = models.IntegerField()
