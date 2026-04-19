from django.db import models
from Experiments.models import VisualiseEnvs, TimestampEnvGenerated
import uuid

class RunMetricsSummary(models.Model):
    run = models.OneToOneField(TimestampEnvGenerated, on_delete=models.CASCADE)

    final_mean_reward = models.FloatField()
    max_mean_reward = models.FloatField()
    mean_dx_final = models.FloatField()

    smoothness_score = models.FloatField(null=True)
    stability_score = models.FloatField(null=True)
    energy_proxy = models.FloatField(null=True)

    convergence_update = models.IntegerField(null=True)
    plateau_detected = models.BooleanField(default=False)

class EnvironmentFitness(models.Model):
    visualise_env = models.ForeignKey(VisualiseEnvs, on_delete=models.CASCADE)
    
    env_index = models.IntegerField()  # env_0, env_1, etc
    json_path = models.CharField(max_length=500)

    fitness_score = models.FloatField()  # std(scores)

    created_at = models.DateTimeField(auto_now_add=True)

class RobotPerformance(models.Model):
    environment = models.ForeignKey(EnvironmentFitness, on_delete=models.CASCADE)

    model_path = models.CharField(max_length=500)

    start_x = models.FloatField()
    max_x = models.FloatField()
    distance_travelled = models.FloatField()
    normalized_score = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

class RunMetricsTimeSeries(models.Model):
    run = models.ForeignKey(TimestampEnvGenerated, on_delete=models.CASCADE)

    update_idx = models.IntegerField()
    global_timestep = models.IntegerField()

    mean_reward = models.FloatField()
    std_reward = models.FloatField()
    mean_dx = models.FloatField()
    mean_speed = models.FloatField()

    entropy = models.FloatField(null=True)
    policy_loss = models.FloatField(null=True)
    value_loss = models.FloatField(null=True)

    class Meta:
        ordering = ["update_idx"]


class DiversityMetric(models.Model):
    run = models.ForeignKey(VisualiseEnvs, on_delete=models.CASCADE, related_name="diversity")

    iou_score = models.FloatField()
    ncd_score = models.FloatField()
    novelty_score = models.FloatField(null=True)

    computed_at = models.DateTimeField(auto_now_add=True)


