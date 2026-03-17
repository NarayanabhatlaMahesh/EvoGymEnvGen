import subprocess, sys

subprocess.Popen(["C:\\redis\\redis-server"])
subprocess.Popen([sys.executable, "-m", "celery", "-A", "TraverseLab", "worker", "-l", "info", "--pool=solo"])
subprocess.call([sys.executable, "manage.py", "runserver"])
