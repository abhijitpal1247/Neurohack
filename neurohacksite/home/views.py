from crypt import methods
from django.shortcuts import render
from django.http import HttpResponse
from .models import Ticketdata
import numpy as np
import plotly.express as px
import pandas as pd
# Create your views here.

def home(request):
     return render(request, 'home/index.html')

def categorization(request):
     Tickets = Ticketdata.objects.all()
     L1_Tag = [i.L1_Tag for i in  Tickets]
     L2_Tag = [i.L2_Tag for i in  Tickets]
     count = np.ones((Ticketdata.objects.count()))
     data = pd.DataFrame({'L1_Tag':L1_Tag,
                         'L2_Tag':L2_Tag,
                         'count':count})

     data['All_Tags'] = 'All Tags'
     data.dropna(inplace=True)
     fig = px.treemap(data, path=["All_Tags", 'L1_Tag', 'L2_Tag'], values='count')
     fig.update_traces(root_color="lightgrey")
     fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
     graph = fig.to_html(full_html=False, default_height=500, default_width=1000)
     context = {'graph': graph}
     return render(request, 'home/categorization.html', context)

def our_team(request):
     return render(request, 'home/our_team.html')

def visualizations(request):
     Tickets = Ticketdata.objects.all()
     L1_Tag = set([i.L1_Tag for i in  Tickets])
     if request.method == "POST":
          print(request.POST.get("L1_Tag"))
     return render(request, 'home/visualizations.html')