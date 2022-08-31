from calendar import c
from crypt import methods
from multiprocessing import context
from turtle import title
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
import json
import sys
from scipy.special import softmax

import matplotlib.pyplot as plt
# Create your views here.

model = Top2Vec.load('home/ml_models/top2vec')
model_reduced = Top2Vec.load('home/ml_models/top2vec_reduced')
obt_topics, obt_topic_score, _ =model.get_topics()
obt_topics_reduced, obt_topic_score_reduced, _ =model_reduced.get_topics()
ano_tickets = pd.DataFrame(list(Trend_data.objects.all().values()))
ano_tickets['Created_date'] = pd.to_datetime(ano_tickets["Created_time"]).dt.strftime('%Y-%m-%d')
ano_tickets['Created_month'] = pd.to_datetime(ano_tickets["Created_time"]).dt.strftime('%Y-%m')
valid_dates = ano_tickets['Created_date'].unique()
valid_months = ano_tickets['Created_month'].unique()

#print(f" valid dates format : {valid_dates[0]}\n valid months format : {valid_months[0]}")
# print(ano_tickets.columns)

valid_dates = json.dumps(valid_dates.tolist())
valid_months = json.dumps(valid_months.tolist())
#print(valid_months)
#print(valid_dates)






with open('../Results/tag_groups.pkl', 'rb') as file:
    # Call load method to deserialze
    l1_tag_list = pickle.load(file)
with open('../Results/tag_names.pkl', 'rb') as file:
    # Call load method to deserialze
    tag_names = pickle.load(file)
    l1_tags = tag_names['l1_tags']
    l2_tags = tag_names['l2_tags']

Tickets = Ticketdata.objects.all()
L1_Tag = [i.L1_Tag for i in  Tickets]
L2_Tag=[i.L2_Tag for i in  Tickets]

L1_to_L2_tags = {}
for i in range(len(l1_tag_list)):
     l2_temp = []
     for j in l1_tag_list[i]:
          l2_temp.append(l2_tags[j])
     L1_to_L2_tags[l1_tags[i]] = set(l2_temp)

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

# Tickets = Ticketdata.objects.all()
# l1_worcloud = {}
# l2_wordcloud = {}
# def generate_wordcloud_fig(wordcloud_image):
#      fig = px.imshow(wordcloud_image)
#      fig.update_layout(
#           xaxis={'visible': False},
#           yaxis={'visible': False},
#           margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
#           hovermode=False,
#           paper_bgcolor="#F9F9FA",
#           plot_bgcolor="#F9F9FA",
#      )
#      return fig

# def generate_wordcloud_div(input_df):
#      #print(input_df)
#      input_df = [preprocess_text(i) for i in input_df]
#      #input_df = input_df.apply(lambda row: preprocess_text(row))
#      stopwords_set = set(stopwords.words())
#      words_to_add=['hi','team']
#      stopwords_set.update(words_to_add)
#      stopwords_set.update(STOPWORDS)
#      # instantiate wordcloud
#      wordcloud = WordCloud(
#           stopwords=stopwords_set,
#           min_font_size=8,
#           scale=2.5,
#           background_color='#F9F9FA',
#           collocations=True,
#           regexp=r"[a-zA-z#&]+",
#           max_words=30,
#           min_word_length=4,
#           collocation_threshold=3
#      )

#      # generate image
#      wordcloud_text = " ".join(text for text in input_df)
#      wordcloud_image = wordcloud.generate(wordcloud_text)
#      wordcloud_image = wordcloud_image.to_array()
#      fig = generate_wordcloud_fig(wordcloud_image)
#      return fig.to_html(full_html=False, default_height=300, default_width=600)

# for L1_Tag_input in l1_tags.values():
#      ticket_desc_l1=[i.Short_description for i in  Tickets.filter(L1_Tag=L1_Tag_input)]
#      l1_worcloud[L1_Tag_input] = generate_wordcloud_div(ticket_desc_l1)
#      for L2_Tag_input in l2_tags.values():
#           L2_Tag=set([i.L2_Tag for i in  Tickets.filter(L1_Tag=L1_Tag_input)])
#           ticket_desc_l2=[i.Short_description for i in  Tickets.filter(L2_Tag=L2_Tag_input)]
#           if len(ticket_desc_l2)!=0:
#                l2_wordcloud[L2_Tag_input]= generate_wordcloud_div(ticket_desc_l2)    



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
     if request.method == "POST":
          L1_Tag_input=request.POST.get("L1_Tag")
          L2_Tag_input=request.POST.get("L2_Tag")

          # print(L2_Tag)
          # data = pd.DataFrame({'L1_Tag':L1_Tag,
          #                 'L2_Tag':L2_Tag,
          #                'desc':ticket_desc})
          # data.dropna(inplace=True)
          # # data_queried=data[(data["L1_Tag"]==L1_Tag_input) & (data["L2_Tag"]==L2_Tag_input)]
          # print(data_queried)
          def generate_wordcloud_fig(wordcloud_image, topic):
               fig = px.imshow(wordcloud_image)
               fig.update_layout(
                    xaxis={'visible': False},
                    yaxis={'visible': False},
                    margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                    hovermode=False,
                    paper_bgcolor="#F9F9FA",
                    plot_bgcolor="#F9F9FA",
                    title = f"Wordcloud for {topic}"
               )
               return fig
          
          # def generate_wordcloud_div(input_df):
          #      #print(input_df)
          #      input_df = [preprocess_text(i) for i in input_df]
          #      #input_df = input_df.apply(lambda row: preprocess_text(row))
          #      stopwords_set = set(stopwords.words())
          #      words_to_add=['hi','team']
          #      stopwords_set.update(words_to_add)
          #      stopwords_set.update(STOPWORDS)
          #      # instantiate wordcloud
          #      wordcloud = WordCloud(
          #           stopwords=stopwords_set,
          #           min_font_size=8,
          #           scale=2.5,
          #           background_color='#F9F9FA',
          #           collocations=True,
          #           regexp=r"[a-zA-z#&]+",
          #           max_words=30,
          #           min_word_length=4,
          #           collocation_threshold=3
          #      )

          #      # generate image
          #      wordcloud_text = " ".join(text for text in input_df)
          #      wordcloud_image = wordcloud.generate(wordcloud_text)
          #      wordcloud_image = wordcloud_image.to_array()
          #      fig = generate_wordcloud_fig(wordcloud_image)
          #      return fig.to_html(full_html=False, default_height=300, default_width=600)
          topic_num = list(l1_tags.keys())[list(l1_tags.values()).index(L1_Tag_input)]
          word_score_dict = dict(zip(obt_topics[topic_num],
                                       softmax(obt_topic_score[topic_num])))
          l1_wordcloud = WordCloud(width=600,
                      height=300,
                      background_color="white").generate_from_frequencies(word_score_dict)
          l1_wordcloud = l1_wordcloud.to_array()
          graph1 = generate_wordcloud_fig(l1_wordcloud, L1_Tag_input).to_html(full_html=False, default_height=300, default_width=600)
          if L2_Tag_input!='':
               topic_num_reduced = list(l2_tags.keys())[list(l2_tags.values()).index(L2_Tag_input)]
               word_score_dict_reduced = dict(zip(obt_topics_reduced[topic_num_reduced],
                                       softmax(obt_topic_score_reduced[topic_num_reduced])))
               l2_wordcloud = WordCloud(width=600,
                      height=300,
                      background_color="white").generate_from_frequencies(word_score_dict_reduced)
               l2_wordcloud = l2_wordcloud.to_array()
               graph2 = generate_wordcloud_fig(l2_wordcloud, L2_Tag_input).to_html(full_html=False, default_height=300, default_width=600)
          else:
               graph2=None
          
          context = {'L1_Tag':set(L1_Tag),
                          'L2_Tag':L1_to_L2_tags[L1_Tag_input],
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
     # context ={'day':list(range(1,32)),'month':list(range(1,13)),'year':list(range(2021,2023))}
     context = {'valid_dates':valid_dates,'valid_months':valid_months}
     context = context|{'day_persist':1,'month_persist':1,'year_persist':2021 }
     
     def anomaly_detection(dataframe,column_field,month,year,day=None):
          if column_field=="Resolution time_hrs":
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
               DF_forest=DF[['month','Resolution time_hrs','year','day']]
               DF_forestd=DF_forest.loc[(DF_forest['month']==int(month)) & (DF_forest['year']==int(year))
                              & (DF_forest['day']==int(day))]
               to_model_columns=['Resolution time_hrs']
               # print(DF_forestm.shape)
               clf_day=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.08), \
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
               clf_day.fit(DF_forestd[to_model_columns])
               
               DF_forest=DF[['month','Resolution time_hrs','year']]
               DF_forestm=DF_forest.loc[(DF_forest['month']==int(month)) & (DF_forest['year']==int(year))]
               to_model_columns=['Resolution time_hrs']
               clf_month=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.15), \
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
               clf_month.fit(DF_forestm[to_model_columns])

               #DF_forestm.dropna(inplace=True)
               
               
               # DF_forestm['anomaly']=pd.Series(clf.predict(DF_forestm[to_model_columns])).apply(lambda x: 'yes' if (x == -1) else 'no' )
               #pred = clf.predict(DF_forestm[to_model_columns])
               #DF_forestm['anomaly']=['yes' if i == -1 else 'no' for i in clf.predict(DF_forestm[to_model_columns])]
               # print(clf.predict(DF_forestm[to_model_columns]))
               # print(DF_forestm['anomaly'])
               DF_anomaly_m = DF_forestm.copy()
               DF_anomaly_d = DF_forestd.copy()
               DF_anomaly_m['anomaly_month'] = ['yes' if i == -1 else 'no' for i in clf_day.predict(DF_forestm[to_model_columns])]
               DF_anomaly_d['anomaly_day'] = ['yes' if i == -1 else 'no' for i in clf_month.predict(DF_forestd[to_model_columns])]
               # print(DF_anomaly_d['anomaly_day'])
               # print(DF_anomaly_m['anomaly_month'])
               #print(DF_forestm.columns)
               return DF_anomaly_m,DF_anomaly_d
          elif column_field=="Reassignment count":
               dataframe["Created_time"] = pd.to_datetime(dataframe["Created_time"])
               dataframe.sort_values(by='Created_time',inplace=True)
               DF = dataframe.set_index(dataframe['Created_time'])
               DF.drop('Created_time',axis=1,inplace=True)
               # DF['Resolution time_hrs']=DF['Resolution_time']/60
               DF['month']=pd.DatetimeIndex(DF.index).month
               DF['year']=pd.DatetimeIndex(DF.index).year
               DF['day']=pd.DatetimeIndex(DF.index).day
               DF.dropna(inplace=True)
               print(DF.columns)
               # print(DF.columns)
               #print(DF, "z")
               DF_forest=DF[['month','Reassigment_count','year','day']]
               DF_forestd=DF_forest.loc[(DF_forest['month']==int(month)) & (DF_forest['year']==int(year))
                              & (DF_forest['day']==int(day))]
               to_model_columns=['Reassigment_count']
               # print(DF_forestm.shape)
               clf_day=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.08), \
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
               clf_day.fit(DF_forestd[to_model_columns])
               
               DF_forest=DF[['month','Reassigment_count','year']]
               DF_forestm=DF_forest.loc[(DF_forest['month']==int(month)) & (DF_forest['year']==int(year))]
               to_model_columns=['Reassigment_count']
               clf_month=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.15), \
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
               clf_month.fit(DF_forestm[to_model_columns])

               #DF_forestm.dropna(inplace=True)
               
               
               # DF_forestm['anomaly']=pd.Series(clf.predict(DF_forestm[to_model_columns])).apply(lambda x: 'yes' if (x == -1) else 'no' )
               #pred = clf.predict(DF_forestm[to_model_columns])
               #DF_forestm['anomaly']=['yes' if i == -1 else 'no' for i in clf.predict(DF_forestm[to_model_columns])]
               # print(clf.predict(DF_forestm[to_model_columns]))
               # print(DF_forestm['anomaly'])
               DF_anomaly_m = DF_forestm.copy()
               DF_anomaly_d = DF_forestd.copy()
               DF_anomaly_m['anomaly_month'] = ['yes' if i == -1 else 'no' for i in clf_day.predict(DF_forestm[to_model_columns])]
               DF_anomaly_d['anomaly_day'] = ['yes' if i == -1 else 'no' for i in clf_month.predict(DF_forestd[to_model_columns])]
               DF_anomaly_m.rename(columns = {'Reassigment_count':'Reassignment_count'}, inplace = True)
               DF_anomaly_d.rename(columns = {'Reassigment_count':'Reassignment_count'}, inplace = True)
               # print(DF_anomaly_d['anomaly_day'])
               # print(DF_anomaly_m['anomaly_month'])
               #print(DF_forestm.columns)
               return DF_anomaly_m,DF_anomaly_d
     
     if request.method == "POST":
          # try:
          feature = request.POST.get("selectField")
          print(feature)
          # print(request.POST.get("selectDate"))
          year,month,day = request.POST.get("selectDate").split("-")
          # print(day, month, year)
          df_anomaly_m, df_anomaly_d = anomaly_detection(ano_tickets,feature,month=month,
                                        year=year,day=day)
          # except Exception as e:
          #      print(e)
          #      return render(request, 'home/index.html', context|{'error':"No entries found! Please select a valid date"})
          if feature=="Resolution time_hrs":
               fig1 = px.line(df_anomaly_d.reset_index(), x='Created_time', y='Resolution time_hrs', color='anomaly_day', title=f'Anomaly Detection for {day}/{month}/{year}')

               fig1.update_xaxes(
               rangeslider_visible=True,
               )
               fig1.update_layout(yaxis_range=[-100,int(df_anomaly_d['Resolution time_hrs'].max())+3])
               #fig.show()
               fig1.update_layout(margin = dict(t=50, l=25, r=25, b=25), title_text=f'Anomaly Detection for {day}/{month}/{year}', title_x = 0.5)
               graph1 = fig1.to_html(full_html=False, default_height=450, default_width=1000)

               context = context|{'graph1': graph1}

               fig2 = px.line(df_anomaly_m.reset_index(), x='Created_time', y='Resolution time_hrs', color='anomaly_month', title=f'Anomaly Detection for {month}')

               fig2.update_xaxes(
               rangeslider_visible=True,
               )
               fig2.update_layout(yaxis_range=[-100,int(df_anomaly_m['Resolution time_hrs'].max())+3])
               #fig.show()
               fig2.update_layout(margin = dict(t=50, l=25, r=25, b=25), title_text=f'Anomaly Detection for {month}/{year}', title_x = 0.5)
               graph2 = fig2.to_html(full_html=False, default_height=450, default_width=1000)

               context = context|{'graph2': graph2}

               context = context|{'day_persist':request.POST.get("selectDate"),'feature_persist': request.POST.get("selectField")}
               
               return render(request, 'home/index.html', context)
          elif feature=="Reassignment count":
               fig1 = px.scatter(df_anomaly_d.reset_index(), x='Created_time', y='Reassignment_count', color='anomaly_day', title='Anomaly Detection')

               fig1.update_xaxes(
               rangeslider_visible=True,
               )
               #fig1.update_layout(yaxis_range=[-100,int(df_anomaly_d['Reassignment count'].max())+3])
               #fig.show()
               fig1.update_layout(margin = dict(t=50, l=25, r=25, b=25), title_text=f'Anomaly Detection for {day}/{month}/{year}', title_x = 0.5)
               graph1 = fig1.to_html(full_html=False, default_height=450, default_width=1000)

               context = context|{'graph1': graph1}

               fig2 = px.scatter(df_anomaly_m.reset_index(), x='Created_time', y='Reassignment_count', color='anomaly_month', title='Anomaly Detection')

               fig2.update_xaxes(
               rangeslider_visible=True,
               )
               #fig2.update_layout(yaxis_range=[-100,int(df_anomaly_m['Reassignment count'].max())+3])
               #fig.show()
               fig2.update_layout(margin = dict(t=50, l=25, r=25, b=25), title_text=f'Anomaly Detection for {month}/{year}', title_x = 0.5)
               graph2 = fig2.to_html(full_html=False, default_height=450, default_width=1000)

               context = context|{'graph2': graph2}

               context = context|{'day_persist':request.POST.get("selectDate"),'feature_persist': request.POST.get("selectField")}
               
               return render(request, 'home/index.html', context)
     else:
          return render(request, 'home/index.html',context)
