#%%import
import pandas as pd
from tqdm import tqdm
import os
#%%
tqdm.pandas()
folder = "..//official_data//Tag_Data"
df_data1 = pd.read_excel(os.path.join(folder,'Data_1.xlsx'))
df_data2 = pd.read_excel(os.path.join(folder,'Data_2.xlsx'),sheet_name=1)
#%%
# %% concatenate columns
cols = ['Short description', 'Description', 'Category', 'Subcategory']
df_data1[cols] = df_data1[cols].fillna('')
df_data1['description'] = df_data1[cols].progress_apply(lambda row: '\t'.join(row.values.astype(str)), axis=1)
#%%
df_data2['description'] = df_data2['Description']
df_combined = pd.concat([df_data1[['Number','description']],df_data2[['Number','description']]])
#%%
df_combined.to_csv(os.path.join(folder,'data_combined.csv'),index=False)
# %%
