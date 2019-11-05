#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:25:31 2019

                            TIME SERIES ANALYSIS

@author: antoinepetit

Summary:
    - Data scraping
    - Data preprocessing
    - Exploratory data analysis (description)
    - Time series analysis (explanation)
    - Time series forecasting (prediction)
    
"""
from datetime import datetime, timedelta
from itertools import chain, product
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import seaborn as sns
import statsmodels.api as sm

website = 'http://www.leblogtvnews.com/2017/09/les-audiences-chaque-jour-de-tpmp-quotidien-et-c-a-vous.html'

# -----------------------------------------------------------------------------
# Conversion tables
# -----------------------------------------------------------------------------
class_name = {"txt_data": "ob-section ob-section-html"}

days_of_week = {'Lundi':"Monday",
                'Mardi':"Tuesday",
                'Merc.':"Wednesday",'Mercredi':"Wednesday",
                'Jeudi':"Thursday",
                'Vend.':"Friday",'Vendredi':"Friday"}

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

month_names = {u'JANVIER': 'January',
               u'FEVRIER':'February', u"F\xc9VRIER": 'February',
               u'MARS':'March', 
               u'AVRIL':'April', 
               u'MAI': 'May', 
               u'JUIN':'June', 
               u'JUILLET':'July', 
               u'AOUT':'August', 
               u'SEPTEMBRE':'September', 
               u'OCTOBRE':'October', 
               u'NOVEMBRE':'November',
               u'DECEMBRE':'December'}

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

holidays_dates = [[pd.Timestamp('2017-12-22 00:00:00'), pd.Timestamp('2018-01-08 00:00:00')],
                  [pd.Timestamp('2018-06-29 00:00:00'), pd.Timestamp('2018-09-03 00:00:00')],
                  [pd.Timestamp('2018-12-21 00:00:00'), pd.Timestamp('2019-01-07 00:00:00')]]

# -----------------------------------------------------------------------------
# Data processing functions
# -----------------------------------------------------------------------------
def get_data_each_day(a_string):
    '''
    Split a long strip into a set of strings corresponding to each day in the month
    e.g. [('Lundi 03', 'Partie 1: 0.84 (19h39-20h32) Partie 2: 1.13(20h41-21h22!) Partie 1: 1.06 (19h27-20h07) Partie 2: 1.69 (20h14-21h12) 1.49'), 
         ('Mardi 04', 'Partie 1: 0.91 (19h39-20h29) Partie 2: 1.37 (20h38-21h21!) Partie 1: 1.00 (19h26-20h04) Partie 2: 1.57 (20h11-21h14) 1.54'), 
         ('Merc. 05', 'Partie 1: 0.92 Partie 2: 1.12 (20h38-21h18) Partie 1: 0.97 Partie 2: 1.53 (20h10-21h10) 1.55')
    Keep the delimiter (day name + day number)
    '''
    return zip(re.findall(r'[a-zA-Z]+\s+\d+\d+\s|[a-zA-Z]+\.+\s+\d+\d+\s|[a-zA-Z]+\.+\d+\d+\s',a_string),
               re.split('[a-zA-Z]+\s+\d+\d+\s|[a-zA-Z]+\.+\s+\d+\d+\s|[a-zA-Z]+\.+\d+\d+\s',a_string)[1:])


def treat_audience_data(a_string):
    '''
    Split the daily viewers data into proper subsets
    e.g. [Partie 1 : 0.84   Partie 2 : 1.47 (20h40-21h12) Partie 1 : 0.85 Partie 2 :  1.75  (20h12-21h14)  1.22]
    to [0.84, 1.47, 0.85, 1.75, 1.22]
    '''
    if a_string[:4] == 'Part':
        # Split at Partie
        # e.g. [Partie 1: 0.84 Partie 2: 1.47 (20h40-21h12) Partie 1: 0.85 Partie 2 : 1.75 (20h12-21h14) 1.22]
        
        temp_string = re.findall(r'\d+\.+\d+\d',a_string)
        return temp_string
    
    elif a_string[0].isdigit():
        # If first character is a digit:
        # e.g. [1.14   (20h37-21h15)   1.29 (20h11-21h11)  1.12]
        
        # Remove text '20h30-20h45'
        temp_string_1 = re.sub('\(\d\dh\d\d-\d\dh\d\d\)','',a_string).split()
        # remove remaining '20h30' and other texts 'best-of'
        temp_string_2 = [ss for ss in temp_string_1 if bool(re.search(r'\d', ss)) and not bool(re.search(r'h', ss))]
        # Remove remaining parenthesis
        temp_string_3 = [re.sub('\)|\(|\!','',sss) for sss in temp_string_2]
        return [ii for ii in temp_string_3 if not (ii=='1' or ii=='2')]
    
    elif not bool(re.search(r'\d', a_string)):
        # if there is not digit in the string:
        # e.g. [best-of   best-of]
        
        return [np.nan, np.nan, np.nan, np.nan, np.nan]
    
    else: # if starts with long comment
        temp_string = re.findall(r'\d+\.+\d+\d',a_string)
        return temp_string


def check_audience_format(an_array):
    '''
    Creates duplicate data, when Part 1/Part 2 are not distinguished
    '''
    new_array = an_array[:]
    for idx,line in enumerate(an_array):
        
        if line:
            if line[0] is not np.nan:
                new_line = [re.findall(r'\d+\.+\d+\d', num)[0] for num in line]
            else:
                new_line = line[:]
            
            if len(line)==3:
                new_array[idx]=[np.nan,float(new_line[0]),np.nan,float(new_line[1]),float(new_line[2])]
            elif len(line)==4:
                new_array[idx]=[float(new_line[0]),float(new_line[0]),float(new_line[1]),float(new_line[2]),float(new_line[3])]
            elif len(line)==5:
                new_array[idx]=[float(ii) for ii in new_line]          
            elif len(line)==6:
                new_array[idx]=[float(ii) for ii in new_line[1:]]
            else:
                print new_line
                raise ValueError('Edit check_audience_format function')
        else:
            new_array[idx] = [np.nan, np.nan, np.nan, np.nan, np.nan]
        
    return new_array


def perdelta(start, end, delta):
    ''' 
    Generate times between start and end with time increment delta (without weekends).
    
    Args:
        start (Timestamp):  start time.
        end (Timstamp):  end time.
        delta (timedelta): time increment.        
    '''
    out_list = []
    curr = start.to_pydatetime()  #convert Timestamp to datetime
    out_list.append(curr)
    while curr < end.to_pydatetime():
        curr += delta
        if curr.weekday() not in [5,6]:  # if it is not Saturday or Sunday
            out_list.append(curr)
    return out_list


def add_missing_data(time_series, a_df, days_of_week):
    '''
    Add missing days with no data as np.nan for each row (for plotting purposes).
    '''
    for a_day in time_series:
        if not (a_df['date'].isin([a_day])).any():  # a_df['date'] is a pd.Series type
            row = pd.DataFrame([[a_day,'',a_day.strftime("%A"),np.nan,np.nan,np.nan,np.nan,np.nan]],columns=list(a_df.columns.values))
            a_df = a_df.append(row, ignore_index=True)
            
    return a_df

# -----------------------------------------------------------------------------
# SCRAPE AND FORMAT VIEWERS DATA
# -----------------------------------------------------------------------------
page = requests.get(website)
soup = BeautifulSoup(page.content, 'html.parser')  # Get the html file
class_htmls = {k:soup.find_all('div', class_=v) for (k,v) in class_name.items()}  # each element contains a table (corresponding to a month)
df_temp = pd.DataFrame({k: [class_htmls[k][i].get_text(separator=' ').replace('\n',' ').strip().split('TPMP') for i in range(1,len(class_htmls[k])-3)] for k in class_name.keys()})

# Get the month/year of each row and put them into a different column / the remaining text goes in a 3rd column
df_temp.loc[:, 'month_name'] = df_temp['txt_data'].map(lambda x: re.split('(\d+)',x[0])[0].strip())
df_temp.loc[:, 'year_name'] = df_temp['txt_data'].map(lambda x: re.split('(\d+)',x[0])[1].strip())
df_temp.loc[:, 'text_1'] = df_temp['txt_data'].map(lambda x: x[1].split('VOUS')[1].strip().split('20H')[-1].strip())
df_temp['text_1'] = [df_temp['text_1'][ii].encode('ascii', 'ignore') for ii in range(df_temp.shape[0])]

# Split the info in 'text' per day into multiple columns list items
df_temp['text_1'] = df_temp['text_1'].apply(get_data_each_day)

# REMOVE WRONG ENTRIES (that don't match with a specific date)
new_dates = [[df_temp.loc[row_id].month_name, df_temp.loc[row_id].year_name, [txt[0] for txt in df_temp.loc[row_id].text_1]] for row_id in range(df_temp.shape[0])]
new_dates_2, rem_idx_list = [], []
for month_idx,item in enumerate(new_dates):
    rem_idx_list.append([idx for idx,astring in enumerate(item[-1]) if ''.join([astr for astr in astring if not astr.isdigit()]).strip() not in days_of_week.keys()])
    new_dates_2.append([[datetime.strptime(month_names[item[0]]+' '+str(item[1])+ ' '+str(elt[-3:].strip()),'%B %Y %d'),str(elt[:-3]).strip()] 
                                                                                for ii,elt in enumerate(item[-1]) if ii not in rem_idx_list[month_idx]])

new_info = [[item[1] for it_idx,item in enumerate(df_temp.loc[row_id].text_1) if it_idx not in rem_idx_list[row_id]] for row_id in range(df_temp.shape[0])]
new_info_2 = [item.strip() for sublist in new_info for item in sublist]  # flatten new_info list

# Save data
file_name = str("".join(re.split('-|\s|:',str(datetime.now())))[4:-9])
with open('data_file_' + file_name, 'a') as my_file:
    my_pickler = pickle.Pickler(my_file)
    data_dic = {}
    data_dic['temp dataframe df_temp'] = df_temp
    my_pickler.dump(data_dic)

# -----------------------------------------------------------------------------
# CREATE CLEAN DF WITH USABLE DATA 
# -----------------------------------------------------------------------------
days_list = list(chain.from_iterable(new_dates_2))  # flatten new_days_2 list
df = pd.DataFrame([days_list[kk] for kk in range(len(days_list))],columns=['date','day_name_fr'])
df['day_name_fr'] = df['day_name_fr'].apply(lambda x: x.strip())
df['day_name_eng'] = df['date'].apply(lambda x: x.strftime("%A"))

# check that english day name match with french
if not (df['day_name_fr'].map(days_of_week) == df['day_name_eng']).all():
    print df['day_name_fr'].map(days_of_week) == df['day_name_eng']
    raise ValueError('Wrong day')

# Process and clean viewers values
processed_audiences = np.array(check_audience_format([treat_audience_data(item) for item in new_info_2])).T
df['TPMP Partie 1'],df['TPMP Partie 2'],df['Quotidien Partie 1'],df['Quotidien Partie 2'],df['C A Vous'] = processed_audiences

## Add possible missing days with no data
time_range = perdelta(df['date'].min(),df['date'].max(), timedelta(days=1.))
df = add_missing_data(time_range, df, days_of_week)

df.sort_values('date', inplace=True,ascending=False)  # order chronologically
df = df.reset_index(drop=True)  # reset rows index (and drop the previous one)
#df = df.dropna(thresh=4)  # remove rows than at most 4 non N.A.

# -----------------------------------------------------------------------------
# DATA PLOTTING
# -----------------------------------------------------------------------------
# Split Part 1 and Part 2:
df_part1 = df[['date','day_name_eng','TPMP Partie 1','Quotidien Partie 1','C A Vous']].copy()
df_part2 = df[['date','day_name_eng','TPMP Partie 2','Quotidien Partie 2','C A Vous']].copy()
df_part1['total'] = df_part1[['TPMP Partie 1','Quotidien Partie 1','C A Vous']].sum(axis=1,skipna=False)  # sum columns
df_part2['total'] = df_part2[['TPMP Partie 2','Quotidien Partie 2','C A Vous']].sum(axis=1,skipna=False)
df_part1['month'] = df_part1['date'].apply(lambda x: x.month)
df_part2['month'] = df_part2['date'].apply(lambda x: x.month)

df_part1 = df_part1.rename(columns={'TPMP Partie 1': 'TPMP', 'Quotidien Partie 1': 'Q', 'C A Vous': 'CAV'})
df_part2 = df_part2.rename(columns={'TPMP Partie 2': 'TPMP', 'Quotidien Partie 2': 'Q','C A Vous': 'CAV'})

# for each show, viewers evolution over time
fig1 = plt.figure(1)
ax11 = fig1.add_subplot(211)
df_part1.plot(x='date',y=['TPMP','Q','CAV'], ax=ax11, legend=False)
plt.title('Part 1 viewers')
plt.xticks([])
plt.xlabel('')
ax12 = fig1.add_subplot(212)
df_part2.plot(x='date',y=['TPMP','Q','CAV'], ax=ax12)
plt.title('Part 2 viewers')
plt.legend(loc='center left', bbox_to_anchor=(1., 1.))
plt.xlabel('')

# Plot total viewers over time
#fig2 = plt.figure(2)
#ax2 = fig2.add_subplot(111)
#df_part1.plot(x='date',y='total', ax=ax2, legend=False)
#plt.title('Part 1 total daily viewers')
#for hol_days in holidays_dates:
#    holiday_beg_idx = df_part1[df_part1['date']==hol_days[0]].index[0] - 1
#    holiday_end_idx = df_part1[df_part1['date']==hol_days[1]].index[0] + 1
#    plt.axvspan(df_part1['date'].iloc[holiday_beg_idx], df_part1['date'].iloc[holiday_end_idx], color='grey', alpha=0.15, lw=0)
#    if (hol_days[1]-hol_days[0]).days > 30:
#        plt.text(df_part1['date'].iloc[int((holiday_beg_idx+holiday_end_idx)/2)+5], 3.0, 'HOLIDAYS', rotation = 90., style='italic', size='x-large')
#plt.xlabel('')

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
fig3plt = df_part2.plot(x='date',y='total', ax=ax3, sharex=True, sharey=True, legend=False)
plt.title('Part 2 total daily viewers')
#for hol_days in holidays_dates:
#    holiday_beg_idx = df_part1[df_part1['date']==hol_days[0]].index[0] - 1
#    holiday_end_idx = df_part1[df_part1['date']==hol_days[1]].index[0] + 1
#    plt.axvspan(df_part1['date'].iloc[holiday_beg_idx], df_part1['date'].iloc[holiday_end_idx], color='grey', alpha=0.15, lw=0)
#    if (hol_days[1]-hol_days[0]).days > 30:
#        plt.text(df_part1['date'].iloc[int((holiday_beg_idx+holiday_end_idx)/2)+5], 3.5, 'HOLIDAYS', rotation = 90., style='italic', size='x-large')
#plt.xlabel('')

# Create total viewers column for pdm calculation
for col_idx,column in enumerate(df_part1):
    if col_idx > 1 and col_idx < 5:
        df_part1['share ' +  column] = np.where(df_part1[column] is np.nan, df_part1[column], df_part1[column]/df_part1['total'])

for col_idx,column in enumerate(df_part2):
    if col_idx > 1 and col_idx < 5:
        df_part2['share ' +  column] = np.where(df_part2[column] is np.nan, df_part2[column], df_part2[column]/df_part2['total'])

# for each show, PDM evolution over time
#df_part1.plot.area(x='date',y=['share TPMP','share Q','share CAV'])
#plt.title('Part 1 Market shares')
#plt.xlabel('')
df_part2.plot.area(x='date',y=['share TPMP','share Q','share CAV'])
plt.title('Part 2 Market shares')
plt.xlabel('')

## Group by day of week
fig6 = plt.figure(6)
ax61 = fig6.add_subplot(311)
sns.boxplot(x=df_part2['day_name_eng'], y=df_part2["TPMP"], order=day_order, ax=ax61, showmeans=True, meanline=True)
plt.xlabel('')
plt.ylabel('')
plt.title('TPMP')
ax61.xaxis.grid(False)
plt.ylim([0.2,2.0])
ax62 = fig6.add_subplot(312)
sns.boxplot(x=df_part2['day_name_eng'], y=df_part2["Q"], order=day_order, ax=ax62, showmeans=True, meanline=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Q')
ax62.xaxis.grid(False)
plt.ylim([0.2,2.0])
ax63 = fig6.add_subplot(313)
sns.boxplot(x=df_part2['day_name_eng'], y=df_part2["CAV"], order=day_order, ax=ax63, showmeans=True, meanline=True)
ax63.xaxis.grid(False)
plt.xlabel('')
plt.ylabel('')
plt.title('C A V')
plt.ylim([0.2,2.0])
plt.subplots_adjust(hspace = 0.4)
plt.suptitle("Audience per day of week")

## Group by month
fig7 = plt.figure(7)
ax71 = fig7.add_subplot(311)
sns.boxplot(x=df_part2['month'], y=df_part2["TPMP"], ax=ax71, showmeans=True, meanline=True)
plt.xlabel('')
plt.ylabel('')
plt.title('TPMP')
ax71.xaxis.grid(False)
ax71.set_xticklabels(month_order)
plt.ylim([0.2,2.0])
ax72 = fig7.add_subplot(312)
sns.boxplot(x=df_part2['month'], y=df_part2["Q"], ax=ax72, showmeans=True, meanline=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Q')
ax72.xaxis.grid(False)
ax72.set_xticklabels(month_order)
plt.ylim([0.2,2.0])
ax73 = fig7.add_subplot(313)
sns.boxplot(x=df_part2['month'], y=df_part2["CAV"], ax=ax73, showmeans=True, meanline=True)
ax73.xaxis.grid(False)
ax73.set_xticklabels(month_order)
plt.xlabel('')
plt.ylabel('')
plt.title('C A V')
plt.ylim([0.2,2.0])
plt.subplots_adjust(hspace = 0.4)
plt.suptitle("Audience per month")

# -----------------------------------------------------------------------------
# DATA ANALYSIS
# -----------------------------------------------------------------------------
## Linear regression on total viewers evolution with time since Jan 2019
df_y = df_part2[df_part2['date']>datetime(2019,1,1)][['date','total']]
df_y = df_y.dropna(axis=0)
Y = np.reshape(df_y['total'].values,[df_y.shape[0],1])
X = np.reshape(np.arange(df_part2.shape[0]-len(Y),df_part2.shape[0]),[len(Y),1])
reg = LinearRegression().fit(X, Y)
a, b = reg.coef_, reg.intercept_
Y_check = a*X + b
Y_check = np.concatenate((Y_check,np.empty([df_part2.shape[0]-len(Y),1])*np.nan))
df_part2['short-term'] = Y_check

## Linear regression on total viewers evolution with time since Jan 2018
df_y = df_part2[df_part2['date']>datetime(2018,1,1)][['date','total']]
df_y = df_y.dropna(axis=0)
Y = np.reshape(df_y['total'].values,[df_y.shape[0],1])
X = np.reshape(df_y.index.tolist(),[len(df_y.index),1])
reg = LinearRegression().fit(X, Y)
a, b = reg.coef_, reg.intercept_
Y_check = a*X + b
df_y['long-term'] = Y_check
df_part2 = pd.concat([df_part2, df_y], axis=1)[['date','day_name_eng','TPMP','Q','CAV','total','month','short-term','long-term']]
df_part2=df_part2.iloc[:,~df_part2.columns.duplicated()].copy()

plt.figure(3)
df_part2.plot(x='date',y='short-term',ax=fig3plt, sharex=True, sharey=True, legend=False)
df_part2.plot(x='date',y='long-term',ax=fig3plt, sharex=True, sharey=True, legend=False)

# Autocorrelation
'''
Autocorrelation plots are often used for checking randomness in time series. 
#This is done by computing autocorrelations for data values at varying time lags. 
#If time series is random, such autocorrelations should be near zero for any and 
#all time-lag separations. If time series is non-random then one or more of the 
#autocorrelations will be significantly non-zero. The horizontal lines displayed 
#in the plot correspond to 95% and 99% confidence bands.
'''
plt.figure(8)
pd.plotting.autocorrelation_plot(df_part2['TPMP'].dropna().values.tolist())

# Trend/Seasonality analysis
# The additive model is Y[t] = T[t] + S[t] + e[t]
y = df_part2[['date','TPMP']]
y.loc[:,'date'] = y['date'].apply(lambda x: x.date()).copy()
y = y.set_index('date')
y.index = pd.to_datetime(y.index)
y.resample('B')
y_nona = y.dropna()
decomposition=sm.tsa.seasonal_decompose(y_nona, model='additive', freq=5)
fig = decomposition.plot()
plt.show()

# -----------------------------------------------------------------------------
# TIME SERIES FORECASTING
# -----------------------------------------------------------------------------
'''
ARIMA, short for ‘Auto Regressive Integrated Moving Average’ is actually a 
class of models that ‘explains’ a given time series based on its own past 
values, that is, its own lags and the lagged forecast errors, so that equation 
can be used to forecast future values.
--> linear time series model
'''
y_reverse = y.iloc[::-1]
freq = pd.infer_freq(y_reverse.index)
y_reverse.index.freq = pd.tseries.frequencies.to_offset(freq)
test_start_date = '2019-04-01'

p = d = q = range(0,2)
pdq = list(product(p,d,q))
seasonal_pdq = [(x[0],x[1],x[2],5) for x in list(product(p,d,q))]

# use a “grid search” to find the optimal set of parameters that yields the best performance for our model
min_val = 0.0
params = None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_reverse[:test_start_date],order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit(disp=False)
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            if results.aic<min_val:
                min_val = results.aic
                params = param, param_seasonal
        except:
            continue

print('Optimal ARIMA{}x{} - AIC:{}'.format(params[0], params[1], min_val))
mod = sm.tsa.statespace.SARIMAX(y_reverse[:test_start_date],
                                order=params[0],
                                seasonal_order=params[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=False)
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()

# Forecasting validation
pred = results.get_prediction(start=pd.to_datetime(test_start_date), end=pd.to_datetime('2019-05-28'), dynamic=False)
pred_ci = pred.conf_int()  # default alpha = .05 returns a 95% confidence interval
ax = y_reverse.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('TPMP viewers')
plt.legend()
plt.show()
plt.title('Out of sample prediction')

# Estimator measures
y_forecasted = np.reshape(pred.predicted_mean.values,[len(pred.predicted_mean.values),1])
y_truth = y_reverse[test_start_date:].values
mfe = (y_forecasted - y_truth).mean()
print('The Mean Forecast Error is {}'.format(round(mfe, 2)))

mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

# Produce a new forecast
mod = sm.tsa.statespace.SARIMAX(y_reverse,
                                order=(1, 0, 1),
                                seasonal_order=(0, 1, 1, 5),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=False)
pred_uc = results.get_forecast(steps=10)
pred_ci = pred_uc.conf_int()  # default alpha = .05 returns a 95% confidence interval
ax = y_reverse.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('TPMP viewers')
plt.legend()
plt.show()
plt.title('Forecast for June')
