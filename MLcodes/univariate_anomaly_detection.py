#%%
import pandas as pd
from statsmodels.tsa.stattools import adfuller
#%%
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
#%%
df = pd.read_excel('..\official_data\Ticket Trend Data\Trend_Data_1.xlsx')
df.head()
#%%
df['Created_year'] = pd.DatetimeIndex(df['Created']).year
df['Created_month'] = pd.DatetimeIndex(df['Created']).month
#%%
#%%
df_test = df[['Created_year','Created_month', 'Reassignment count']]
df_year_2021 = df_test.loc[df_test['Created_year']==2021]
df_temp = df_year_2021.groupby([df_year_2021.Created_month]).sum()
#%%
df_month = df_temp.loc[df_temp.index.Created_year==2021]
#%%
test_stationarity(df_test, 'Reassignment count')
# %%
max_p, max_q = 5, 5 
def get_order_sets(n, n_per_set) -> list:
    n_sets = [i for i in range(n)]
    order_sets = [
        n_sets[i:i + n_per_set]
        for i in range(0, n, n_per_set)
    ]
    return order_sets
#%%
def find_aic_for_model(data, p, q, model, model_name):
    try:
        msg = f"Fitting {model_name} with order p, q = {p, q}n"
        print(msg)
        if p == 0 and q == 1:
            # since p=0 and q=1 is already
            # calculated
            return None, (p, q)
        ts_results = model(data, order=(p, q)).fit(disp=False)
        curr_aic = ts_results.aic
        return curr_aic, (p, q)
    except Exception as e:
        f"""Exception occurred continuing {e}"""
        return None, (p, q)
#%%
from joblib import Parallel,delayed
from statsmodels.tsa.arima.model import ARIMA
def find_best_order_for_model(data, model, model_name):
    p_ar, q_ma = max_p, max_q
    final_results = []
    ts_results = model(data, order=(0, 0, 1)).fit()
    min_aic = ts_results.aic
    final_results.append((min_aic, (0, 1)))
    # example if q_ma is 6
    # order_sets would be [[0, 1, 2, 3, 4], [5]]
    order_sets = get_order_sets(q_ma, 5)
    for p in range(0, p_ar):
        for order_set in order_sets:
            # fit the model and find the aic
            results = Parallel(n_jobs=len(order_set), prefer='threads')(
                delayed(find_aic_for_model)(data, p, q, model, model_name)
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
    df_test['Reassignment count'], ARIMA, "ARMA"
)
print(min_aic, min_order)
# %%
def find_anomalies(squared_errors):
    threshold = np.mean(squared_errors) + np.std(squared_errors)
    predictions = (squared_errors >= threshold).astype(int)
    return predictions, threshold