#%% RUN IN django ENV
import pandas as pd, os
from sqlalchemy import create_engine
#%%
# Credentials to database connection
hostname="127.0.0.1"
dbname="neurohack_db"
uname="sandra"
pwd="password"

# Create dataframe


# Create SQLAlchemy engine to connect to MySQL Database
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}?charset=utf8mb4"
				.format(host=hostname, db=dbname, user=uname, pw=pwd))

# %%
# folder = "..//official_data//Tag_Data"
# wb = pd.read_excel(os.path.join(folder,"Sample_data for L1 and L2 tags.xlsx"))
#%%
wb = pd.read_csv('../Results/complete_named_tags_upload.csv')
#%% check existing columns
db_tab_cols = pd.read_sql("select * from home_ticketdata where 1=2", engine).columns.tolist()
#%%
wb = wb[['Number','L1_Tag','L2_Tag','Short description']]
wb['Short description'] = wb['Short description'].str.encode('unicode_escape')                            
wb.columns = db_tab_cols[1:]
#%% Convert dataframe to sql table 
#    

wb.to_sql('home_ticketdata', engine,if_exists='append',index=False )
# %%

# %% anomaly 
# folder = "..//official_data//Tag_Data"
# wb = pd.read_excel(os.path.join(folder,"Sample_data for L1 and L2 tags.xlsx"))
#%%
wb = pd.read_excel('../official_data/Ticket Trend Data/Trend_Data_1.xlsx')
#%%
wb_copy = wb.copy()
#%% check existing columns
db_tab_cols = pd.read_sql("select * from home_trend_data where 1=2", engine).columns.tolist()
#%%
wb = wb[['Number','Created','Resolution time']]
# wb['Short description'] = wb['Short description'].str.encode('unicode_escape')                            
wb.columns = db_tab_cols[1:]
#%% Convert dataframe to sql table 
#    

wb.to_sql('home_trend_data', engine,if_exists='append',index=False )
# %%