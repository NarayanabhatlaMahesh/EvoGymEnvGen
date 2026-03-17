from django.shortcuts import render
import zlib

# Create your views here.
def show_metrics(req):
    return render(req,'metrics/metrics.html')
