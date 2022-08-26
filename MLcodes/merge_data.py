#%%import
import pandas as pd
from tqdm import tqdm
#%%
tqdm.pandas()
df_data1 = pd.read_excel('..//official_data//Tag_Data//Data_1.xlsx')
df_data2 = pd.read_excel('..//official_data//Tag_Data//Data_2.xlsx',sheet_name=1)
#%%
# %% concatenate columns
cols = ['Short description', 'Description', 'Category', 'Subcategory']
df_data1[cols] = df_data1[cols].fillna('')
df_data1['description'] = df_data1[cols].progress_apply(lambda row: '\t'.join(row.values.astype(str)), axis=1)
#%%
df_data2['description'] = df_data2['Description']
df_combined = pd.concat([df_data1[['Number','description']],df_data2[['Number','description']]])
#%%
df_combined.to_csv('official_data//Tag_Data//data_combined.csv',index=False)
# %%
