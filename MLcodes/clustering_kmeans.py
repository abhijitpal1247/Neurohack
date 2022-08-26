#%%
import pandas as pd
from preprocessing import contraction_expander
from tqdm import tqdm

#%%
tqdm.pandas()
#%%
df_combined = pd.read_csv('..\official_data\Tag_Data\data_combined.csv')
df_combined.head()
#%%

df_combined.dropna(inplace=True)
df_combined['preprocessed'] = df_combined.progress_apply(lambda row: contraction_expander(row['description']), axis=1)
df_combined['preprocessed'] = df_combined.progress

# %%
