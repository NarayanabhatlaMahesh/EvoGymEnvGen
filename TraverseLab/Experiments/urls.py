from django.urls import path
from .views import runs as views
from .views import visualize as visviews

urlpatterns = [
    path('',views.redirect_main,name="mainPage"),
    path("runs/", views.run_list, name="run_list"),
    path("runs/<uuid:run_id>/", views.run_detail, name="run_detail"),
    path("experiment/",views.run_GA_env,name="run_GA_env"),
    path("visualise_images/",visviews.env_images,name="visualise_env")
]