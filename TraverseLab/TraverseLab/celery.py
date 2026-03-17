import os
from celery import Celery
import logging
import traceback
from celery.signals import task_failure


os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "TraverseLab.settings"
)

app = Celery("TraverseLab")

app.config_from_object(
    "django.conf:settings",
    namespace="CELERY"
)

app.autodiscover_tasks()
@task_failure.connect
def handle_task_failure(sender, task_id, exception, args, kwargs, traceback_obj, einfo, **extra):
    """
    Triggers on task failure and prints the full Python traceback
    (including File, Line Number, and Function) to the console.
    """
    print("\n" + "!"*80)
    print(f"DETAILED CRASH REPORT: {sender.name}")
    print(f"TASK ID: {task_id}")
    print("-" * 80)
    
    # einfo.traceback contains the full 'last line' style report you want
    # including the file paths and line numbers.
    if einfo:
        print(einfo.traceback)
    else:
        # Fallback: manually print if einfo isn't available
        traceback.print_exception(type(exception), exception, traceback_obj)

    print("!"*80 + "\n")