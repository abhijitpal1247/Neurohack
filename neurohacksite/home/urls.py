from django.urls import path
from . import views
urlpatterns = [
    path('', views.anomaly_vizualization,name="neurohack-anomaly-visualizations"),
    # path('', views.home,name="neurohack-home"),
    path('categorization/', views.categorization,name="neurohack-categorization"),
    path('ourteam/', views.our_team,name="neurohack-our_team"),
    path('visualizations/', views.visualizations,name="neurohack-visualizations"),
]