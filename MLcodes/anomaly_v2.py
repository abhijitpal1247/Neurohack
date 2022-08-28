# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
df=pd.read_excel('../official_data/Ticket Trend Data/Trend_Data_1.xlsx')



# %%

df.head()

# %%

type(df['Created'][0])



# %%

df["Created"] = pd.to_datetime(df["Created"])



# %%

df['Created']

# %%
df.head()
# %%
df.sort_values(by='Created',inplace=True)

# %%
# DF = pd.DataFrame()
# DF['value'] = df['Resolution time']
DF = df.set_index(df['Created'])
DF.drop('Created',axis=1,inplace=True)
# %%
DF.head()
# %%
plt.plot(DF['Resolution time'])
plt.gcf().autofmt_xdate()
plt.show()
# %%
DF['Resolution time_hrs']=DF['Resolution time']/60
# %%
plt.plot(DF['Resolution time_hrs'])
plt.gcf().autofmt_xdate()
plt.show()
# %%
DF['month']=pd.DatetimeIndex(DF.index).month
DF['year']=pd.DatetimeIndex(DF.index).year
# %%

# %%
DF_month=DF[(DF['month']==2) & (DF['year']==2021)]
plt.plot(DF_month['Resolution time_hrs'])
plt.gcf().autofmt_xdate()
plt.show()
# %%
DF_updated=DF[['Resolution time_hrs']]
DF_updated.head()

# %%
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
# %%
def test_stationarity(ts_data, column='', signif=0.05, series=False):
    if series:
        adf_test = adfuller(ts_data, autolag='AIC')
    else:
        adf_test = adfuller(ts_data[column], autolag='AIC')
    p_value = adf_test[1]
    if p_value <= signif:
        test_result = "Stationary"
    else:
        test_result = "Non-Stationary"
    return test_result
# %%

# %%
test_stationarity(DF_updated, 'Resolution time_hrs')
# %%
DF_updated.isna().sum()
# %%
import joblib
# %%
max_p, max_q = 5, 5 
def get_order_sets(n, n_per_set) -> list:
    n_sets = [i for i in range(n)]
    order_sets = [
        n_sets[i:i + n_per_set]
        for i in range(0, n, n_per_set)
    ]
    return order_sets
def find_aic_for_model(data, p, q, model, model_name):
    try:
        msg = f"Fitting {model_name} with order p, q = {p, q}n"
        print(msg)
        if p == 0 and q == 1:
            # since p=0 and q=1 is already
            # calculated
            return None, (p, q)
        ts_results = model(data, order=(p, q,0)).fit(disp=False)
        curr_aic = ts_results.aic
        return curr_aic, (p, q)
    except Exception as e:
        f"""Exception occurred continuing {e}"""
        return None, (p, q)
def find_best_order_for_model(data, model, model_name):
    p_ar, q_ma = max_p, max_q
    final_results = []
    ts_results = model(data, order=(0, 1)).fit(disp=False)
    min_aic = ts_results.aic
    final_results.append((min_aic, (0, 1)))
    # example if q_ma is 6
    # order_sets would be [[0, 1, 2, 3, 4], [5]]
    order_sets = get_order_sets(q_ma, 5)
    for p in range(0, p_ar):
        for order_set in order_sets:
            # fit the model and find the aic
            results = joblib.Parallel(n_jobs=len(order_set), prefer='threads')(
                joblib.delayed(find_aic_for_model)(data, p, q, model, model_name)
                for q in order_set
            )
            final_results.extend(results)
    results_df = pd.DataFrame(
        final_results,
        columns=['aic', 'order']
    )
    min_df = results_df[
        results_df['aic'] == results_df['aic'].min()
    ]
    min_aic = min_df['aic'].iloc[0]
    min_order = min_df['order'].iloc[0]
    return min_aic, min_order, results_df
min_aic, min_order, results_df = find_best_order_for_model(
    DF_updated, ARIMA, "ARMA"
)
print(min_aic, min_order)# Output: 1341.1035677173795, (4, 4)
# %%
