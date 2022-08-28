from crypt import methods
from multiprocessing import context
from django.shortcuts import render
from django.http import HttpResponse
from .models import Ticketdata
import numpy as np
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import io
import urllib, base64
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
     L1_Tag = [i.L1_Tag for i in  Tickets]
     # L2_Tag=[i.L2_Tag for i in  Tickets]
     ticket_desc=[i.Short_description for i in Tickets]
     # data = pd.DataFrame({'L1_Tag':L1_Tag,
     #                      'L2_Tag':L2_Tag,
     #                     'desc':ticket_desc})
     # data.dropna(inplace=True)
     if request.method == "POST":
          L1_Tag_input=request.POST.get("L1_Tag")
          L2_Tag=set([i.L2_Tag for i in  Tickets.filter(L1_Tag=L1_Tag_input)])
          ticket_desc_l1=[i.Short_description for i in  Tickets.filter(L1_Tag=L1_Tag_input)]
          
          L2_Tag_input=request.POST.get("L2_Tag")
          
          ticket_desc_l2=[i.Short_description for i in  Tickets.filter(L2_Tag=L2_Tag_input)]

          # print(L2_Tag)
          # data = pd.DataFrame({'L1_Tag':L1_Tag,
          #                 'L2_Tag':L2_Tag,
          #                'desc':ticket_desc})
          # data.dropna(inplace=True)
          # # data_queried=data[(data["L1_Tag"]==L1_Tag_input) & (data["L2_Tag"]==L2_Tag_input)]
          # print(data_queried)
          def generate_wordcloud_fig(wordcloud_image):
               fig = px.imshow(wordcloud_image)
               fig.update_layout(
                    xaxis={'visible': False},
                    yaxis={'visible': False},
                    margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                    hovermode=False,
                    paper_bgcolor="#F9F9FA",
                    plot_bgcolor="#F9F9FA",
               )
               return fig
          
          def generate_wordcloud_div(input_df):
               stopwords_set = set(stopwords.words())
               words_to_add=['hi','team']
               stopwords_set.update(words_to_add)
               stopwords_set.update(STOPWORDS)
               # instantiate wordcloud
               wordcloud = WordCloud(
                    stopwords=stopwords_set,
                    min_font_size=8,
                    scale=2.5,
                    background_color='#F9F9FA',
                    collocations=True,
                    regexp=r"[a-zA-z#&]+",
                    max_words=30,
                    min_word_length=4,
                    collocation_threshold=3
               )

               # generate image
               wordcloud_text = " ".join(text for text in input_df)
               wordcloud_image = wordcloud.generate(wordcloud_text)
               wordcloud_image = wordcloud_image.to_array()
               fig = generate_wordcloud_fig(wordcloud_image)
               return fig.to_html(full_html=False, default_height=300, default_width=600)
          
          graph1 = generate_wordcloud_div(ticket_desc_l1)
          if len(ticket_desc_l2)!=0:
               graph2 = generate_wordcloud_div(ticket_desc_l2)
          else:
               graph2=None
          
          context = {'L1_Tag':set(L1_Tag),
                          'L2_Tag':set(L2_Tag),
                         'L1_Tag_input':L1_Tag_input,
                         'L2_Tag_input':L2_Tag_input,
                         #  'word_cloud':word_cloud
                         'graph1':graph1,
                         'graph2':graph2
                         }
          return render(request, 'home/visualizations.html',context)
          
     else:
          L1_Tag_input = None
          L2_Tag_input = None
          context = {'L1_Tag':set(L1_Tag),
                         # 'L2_Tag':set(L2_Tag),
                         }
          return render(request, 'home/visualizations.html',context)