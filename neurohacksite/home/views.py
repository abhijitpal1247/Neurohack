from crypt import methods
from multiprocessing import context
from django.shortcuts import render
from django.http import HttpResponse
from .models import Ticketdata
from .models import Trend_data
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import IsolationForest
from nltk.corpus import stopwords
import io
import urllib, base64
from top2vec import Top2Vec
from .preprocess import contraction_expander,url_remover,email_remover,removal_html_tags,digit_remover,special_character_removal,stopwords_removal,lemmatize          
from .preprocess import special_character_removal
import pickle
# Create your views here.

model = Top2Vec.load('home/ml_models/top2vec')
ano_tickets = pd.DataFrame(list(Trend_data.objects.all().values()))
#print(ano_tickets.columns)



with open('../Results/tag_groups.pkl', 'rb') as file:
    # Call load method to deserialze
    l1_tag_list = pickle.load(file)
with open('../Results/tag_names.pkl', 'rb') as file:
    # Call load method to deserialze
    tag_names = pickle.load(file)
    l1_tags = tag_names['l1_tags']
    l2_tags = tag_names['l2_tags']

def home(request):
     return render(request, 'home/index.html')

def preprocess_text(sample):
    sample=contraction_expander(sample)
    sample=url_remover(sample)
    sample=email_remover(sample)
    sample=(removal_html_tags(sample))
    sample=digit_remover(sample)
    sample=special_character_removal(sample)
    sample=stopwords_removal(sample)
    sample=lemmatize(sample)
    return sample

def get_tags(df):
     df['preprocessed_text'] = df.apply(lambda row: preprocess_text(row['Description']),axis=1)
     df.dropna(inplace=True)
     df['L2_tags'] = [model.query_topics(i_str,1)[3][0] for i_str in df['preprocessed_text']]

     l2_to_l1_dict = {}
     for i in range(len(l1_tag_list)):
          for j in l1_tag_list[i]:
               l2_to_l1_dict[j] = i
     df['L1_tags'] = df.apply(lambda row: l2_to_l1_dict[int(row['L2_tags'])], axis=1)
     df['L1_Tag'] = df.apply(lambda row: l1_tags[row['L1_tags']], axis=1)
     df['L2_Tag'] = df.apply(lambda row: l2_tags[row['L2_tags']], axis=1)
     return df

     

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
     fig.update_layout(margin = dict(t=50, l=25, r=25, b=25), title_text='Categorization of tickets into L1 and L2 tags', title_x = 0.5)
     graph = fig.to_html(full_html=False, default_height=500, default_width=1000)
     if request.method == 'POST':

          csv_file = request.FILES["csv_file"]
          #file_data = csv_file.read().decode("utf-8")
          df = pd.read_csv(csv_file)
          # print(df.columns)
          df = df[['Description']]
          df.dropna(inplace=True)
          df = get_tags(df)
          df1 = df[['Description','L1_Tag', 'L2_Tag']]
          df2 = pd.DataFrame(df1)
          #print(type(df1))
          fig = go.Figure(data=[go.Table(    header=dict(values=list(df2.columns),                
                                             fill_color='paleturquoise',                
                                             align='left'),    
                                             cells=dict(values=[df2.Description, df2.L1_Tag, df2.L2_Tag],
                                             fill_color='lavender',              
                                             align='left'))])
          fig.update_layout(margin = dict(t=50, l=25, r=25, b=25), title_text='Ticket Categorization', title_x = 0.5)
          table = fig.to_html(full_html=False, default_height=500, default_width=1000)

          desc = {'table': table, 'graph': graph}
          return render(request, 'home/categorization.html', desc)
            
     
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
               #print(input_df)
               input_df = [preprocess_text(i) for i in input_df]
               #input_df = input_df.apply(lambda row: preprocess_text(row))
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


# anomaly detection
def anomaly_vizualization(request):
     context ={'day':list(range(1,32)),'month':list(range(1,13)),'year':list(range(2021,2023))}
     context = context|{'day_persist':1,'month_persist':1,'year_persist':2021 }
     def anomaly_detection(dataframe,month,year,day=None):
          dataframe["Created_time"] = pd.to_datetime(dataframe["Created_time"])
          dataframe.sort_values(by='Created_time',inplace=True)
          DF = dataframe.set_index(dataframe['Created_time'])
          DF.drop('Created_time',axis=1,inplace=True)
          DF['Resolution time_hrs']=DF['Resolution_time']/60
          DF['month']=pd.DatetimeIndex(DF.index).month
          DF['year']=pd.DatetimeIndex(DF.index).year
          DF['day']=pd.DatetimeIndex(DF.index).day
          DF.dropna(inplace=True)
          #print(DF, "z")
          if day is not None:
               DF_forest=DF[['month','Resolution time_hrs','year','day']]
               DF_forestm=DF_forest.loc[(DF_forest['month']==int(month)) & (DF_forest['year']==int(year))
                             & (DF_forest['day']==int(day))]
               to_model_columns=['Resolution time_hrs']
               # print(DF_forestm.shape)
               clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.08), \
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
               clf.fit(DF_forestm[to_model_columns])
          else:
               DF_forest=DF[['month','Resolution time_hrs','year']]
               DF_forestm=DF_forest.loc[(DF_forest['month']==month) & (DF_forest['year']==year)]
               to_model_columns=['Resolution time_hrs']
               clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.15), \
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
               clf.fit(DF_forestm[to_model_columns])

          #DF_forestm.dropna(inplace=True)
          
          
          # DF_forestm['anomaly']=pd.Series(clf.predict(DF_forestm[to_model_columns])).apply(lambda x: 'yes' if (x == -1) else 'no' )
          #pred = clf.predict(DF_forestm[to_model_columns])
          #DF_forestm['anomaly']=['yes' if i == -1 else 'no' for i in clf.predict(DF_forestm[to_model_columns])]
          # print(clf.predict(DF_forestm[to_model_columns]))
          # print(DF_forestm['anomaly'])
          DF_anomaly = DF_forestm.copy()
          DF_anomaly['anomaly'] = ['yes' if i == -1 else 'no' for i in clf.predict(DF_forestm[to_model_columns])]
          #print(DF_forestm.columns)
          return DF_anomaly
     
     if request.method == "POST":
          try:
               df_anomaly = anomaly_detection(ano_tickets,request.POST.get("month"),
                                         request.POST.get("year"),request.POST.get("day") )
          except:
               return render(request, 'home/index.html', context|{'error':"No entries found! Please select a valid date"})
          
          fig = px.line(df_anomaly.reset_index(), x='Created_time', y='Resolution time_hrs', color='anomaly', title='Anomaly Detection')

          fig.update_xaxes(
          rangeslider_visible=True,
          )
          fig.update_layout(yaxis_range=[-100,int(df_anomaly['Resolution time_hrs'].max())+3])
          #fig.show()
          fig.update_layout(margin = dict(t=50, l=25, r=25, b=25), title_text='Categorization of tickets into L1 and L2 tags', title_x = 0.5)
          graph = fig.to_html(full_html=False, default_height=500, default_width=1000)

          context = context|{'graph': graph}
          context = context|{'day_persist':request.POST.get("day"),'month_persist':request.POST.get("month"),'year_persist':request.POST.get("year") }
          
          return render(request, 'home/index.html', context)
     else:
          return render(request, 'home/index.html',context)



     
