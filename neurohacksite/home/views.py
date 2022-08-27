from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def home(request):
     return render(request, 'home/index.html')

def categorization(request):
     return render(request, 'home/categorization.html')

def our_team(request):
     return render(request, 'home/our_team.html')

def visualizations(request):
     return render(request, 'home/visualizations.html')