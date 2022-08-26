#%%
from preprocessing import contraction_expander,url_remover,email_remover,removal_html_tags,digit_remover,special_character_removal,stopwords_removal,lemmatize,noun_exctraction
                                
from preprocessing import special_character_removal
import pandas as pd
import numpy as np
from tqdm import tqdm


#%%
df=pd.read_csv('..\official_data\Tag_Data\data_combined.csv')
# %%
df.head()
# %%
text=df['description']
# %%
text
# %%
sample=text[246831]


# %%
def preprocess(sample):
    sample=contraction_expander(sample)
    sample=url_remover(sample)
    sample=email_remover(sample)
    sample=(removal_html_tags(sample))
    sample=digit_remover(sample)
    sample=special_character_removal(sample)
    sample=stopwords_removal(sample)
    sample=lemmatize(sample)
    #sample=noun_exctraction(sample)

    return sample
# %%
print(preprocess(sample))
# %%
print(sample)
# %%
tqdm.pandas()

# %%

# %%
df_processed = pd.DataFrame()
#%%
df.dropna(inplace=True)
df['cleaned_desc']=df.progress_apply(lambda row: preprocess(row['description']),axis=1)
# %%
