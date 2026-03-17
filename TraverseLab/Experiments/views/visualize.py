import os
from django.shortcuts import render, get_object_or_404, redirect
from Experiments.models import Run
from Metrics.models import RunMetricsSummary
from Artifacts.models import Artifact
from .. import EvolutionJob
from django.http import JsonResponse
from celery import shared_task
from ..models import Run, VisualiseEnvs, TimestampEnvGenerated
from ..services.evo_terrain_ea import EvoGymTerrainEA

import os
from collections import defaultdict
from django.shortcuts import render



import os
from django.shortcuts import render

def env_images(request):
    images = TimestampEnvGenerated.objects.all()
    for i in images:
        print(i.curriculums.all())
    return render(
        request,
        "experiments/visualise.html",
        {
            "Timestamps": images,
        },
    )




