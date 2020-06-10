
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[43]:

import numpy as np
import pandas as pd


from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
scaler = MinMaxScaler()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def blight_model():



    data = pd.read_csv(filepath_or_buffer='train.csv',engine='python')
    test = pd.read_csv(filepath_or_buffer='test.csv',engine='python')
    address = pd.read_csv(filepath_or_buffer='addresses.csv',engine='python')
    latlon = pd.read_csv(filepath_or_buffer='latlons.csv',engine='python')

    columns_to_keep = list(test.columns)
    columns_to_keep.append('compliance')
    data_select = data[columns_to_keep]

    data_select_1 = data_select[(data_select['compliance']==1) | (data_select['compliance']==0) ]

    to_remove=[]
    for i in range(len(data_select_1.columns)):
        distinct = data_select_1.iloc[:,i].nunique()
        names = data_select_1.columns[i]     
        #print(names,":",distinct) 
        if distinct<5:
            to_remove.append(names)
    to_remove.remove('compliance')

    data_select_2 =  data_select_1.merge(address).merge(latlon)
    data_select_2['distance']= (((-83.238943-data_select_2['lon'])**2)+((42.377249-data_select_2['lat'])**2))**(0.5)
    data_select_2['ticket_issued_date'] = pd.to_datetime(data_select_2['ticket_issued_date'],format='%Y-%m-%d %H:%M:%S')
    data_select_2['hearing_date'] =pd.to_datetime(data_select_2['hearing_date'],format='%Y-%m-%d %H:%M:%S')
    fix_dt = datetime.strptime('2011-12-31','%Y-%m-%d')
    data_select_2['issue_diff'] = (data_select_2['ticket_issued_date']-fix_dt).dt.days
    data_select_2['hearing_diff'] = (data_select_2['hearing_date']-fix_dt).dt.days
    data_select_2 =  data_select_2.drop(to_remove,axis=1)
    data_select_2.drop(['country','address','lat','lon','ticket_id','violation_description','hearing_date','ticket_issued_date'],axis=1,inplace=True)

    data_select_2[['violation_street_number','mailing_address_str_number']] =data_select_2[['violation_street_number','mailing_address_str_number']].astype(str)


    data_select_2['violator_name'] = data_select_2['violator_name'].fillna('Blank')
    data_select_2['mailing_address_str_name'] = data_select_2['mailing_address_str_name'].fillna('Blank')
    data_select_2['state'] = data_select_2['state'].fillna('MI')
    data_select_2['zip_code'] = data_select_2['zip_code'].fillna('0')
    data_select_2['hearing_diff'] = data_select_2['hearing_diff'].fillna((data_select_2['hearing_diff'].mean()))
    data_select_2['distance'] = data_select_2['distance'].fillna((data_select_2['distance'].mean()))


    data_select_2['distance_scaled'] = scaler.fit_transform(data_select_2[['distance']])
    data_select_2['issue_diff_scaled'] = scaler.fit_transform(data_select_2[['issue_diff']])
    data_select_2['hearing_diff_scaled'] = scaler.fit_transform(data_select_2[['hearing_diff']])

    data_select_3  =  data_select_2.drop(['hearing_diff','issue_diff','distance'],axis=1)
    data_select_3 = data_select_3[['agency_name', 'inspector_name', 'violator_name',
           'violation_street_number', 'violation_street_name',
           'mailing_address_str_number', 'mailing_address_str_name', 'city',
           'state', 'zip_code', 'violation_code', 'fine_amount', 'late_fee',
           'discount_amount', 'judgment_amount',  'distance_scaled', 'issue_diff_scaled',
           'hearing_diff_scaled', 'compliance']]

    data_select_4  =  data_select_3.drop(['inspector_name','violator_name','violation_street_name','mailing_address_str_number',
                                          'violation_street_number','mailing_address_str_name','zip_code','city','state',
                                          'violation_code'],axis=1)
    data_select_4=data_select_4.join(pd.get_dummies(data_select_4['agency_name']))
    data_select_4=data_select_4.drop(['agency_name','distance_scaled','issue_diff_scaled','hearing_diff_scaled'],axis=1)
    data_select_4_notarget = data_select_4.drop(['compliance'],axis=1)
    data_select_4_target = data_select_4['compliance']

    X_train, X_test, y_train, y_test = train_test_split(data_select_4_notarget,data_select_4_target, random_state = 3)
    clf = RandomForestClassifier().fit(X_train, y_train)
    clf.score(X_test, y_test)

    test_1 = test[['ticket_id', 'ticket_issued_date', 'hearing_date', 'agency_name',   'fine_amount',
       'late_fee', 'discount_amount'     , 'judgment_amount']]

    test_2 =  test_1.merge(address).merge(latlon)
    test_2['distance']= (((-83.238943-test_2['lon'])**2)+((42.377249-test_2['lat'])**2))**(0.5)
    test_2['ticket_issued_date'] = pd.to_datetime(test_2['ticket_issued_date'],format='%Y-%m-%d %H:%M:%S')
    test_2['hearing_date'] =pd.to_datetime(test_2['hearing_date'],format='%Y-%m-%d %H:%M:%S')
    fix_dt = datetime.strptime('2016-12-31','%Y-%m-%d')
    test_2['issue_diff'] = (test_2['ticket_issued_date']-fix_dt).dt.days
    test_2['hearing_diff'] = (test_2['hearing_date']-fix_dt).dt.days
    test_2['hearing_diff'] = test_2['hearing_diff'].fillna((test_2['hearing_diff'].mean()))
    test_2['distance'] = test_2['distance'].fillna((test_2['distance'].mean()))
    test_2['distance_scaled'] = scaler.fit_transform(test_2[['distance']])
    test_2['issue_diff_scaled'] = scaler.fit_transform(test_2[['issue_diff']])
    test_2['hearing_diff_scaled'] = scaler.fit_transform(test_2[['hearing_diff']])
    test_2=test_2.join(pd.get_dummies(test_2['agency_name']))
    test_2.drop(['address','lat','lon','ticket_id','hearing_date','distance','ticket_issued_date','hearing_diff','issue_diff',
            'agency_name','distance_scaled','issue_diff_scaled','hearing_diff_scaled'],axis=1,inplace=True)

    test_2['Health Department']=0
    test_2['Neighborhood City Halls']=0
    prediction = clf.predict_proba(test_2)

    pred1=[]
    for i in range(len(prediction)):
        pred1.append(prediction[:][i][1])

    pred1 = pd.Series(pred1)

    answer = pred1.to_frame().join(test['ticket_id'])


    answer = answer.set_index('ticket_id')
    
    answer.columns=['compliance']

   
    return answer['compliance']# Your answer here


# In[44]:



blight_model()

