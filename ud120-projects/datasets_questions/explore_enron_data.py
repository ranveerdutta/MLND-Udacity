#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print(len(enron_data))

print(enron_data['METTS MARK'])

print(enron_data['METTS MARK']['poi'])

poi_count = 0

for key, value in enron_data.iteritems():
    if(value['poi'] == True):
        poi_count = poi_count + 1

print(poi_count)


print(enron_data['PRENTICE JAMES']['total_stock_value'])

print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])

print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

if(enron_data['LAY KENNETH L']['total_payments'] > enron_data['SKILLING JEFFREY K']['total_payments']):
    if(enron_data['LAY KENNETH L']['total_payments'] > enron_data['FASTOW ANDREW S']['total_payments']):
        print('LAY KENNETH L')
        print(enron_data['LAY KENNETH L']['total_payments'])
    else:
        print('PRENTICE JAMES')
        print(enron_data['PRENTICE JAMES']['total_payments'])
else:
    if(enron_data['SKILLING JEFFREY K']['total_payments'] > enron_data['FASTOW ANDREW S']['total_payments']):
        print('SKILLING JEFFREY K')
        print(enron_data['SKILLING JEFFREY K']['total_payments'])
    else:
        print('FASTOW ANDREW S')
        print(enron_data['FASTOW ANDREW S']['total_payments'])


valid_email_count = 0
valid_salary_count = 0
total_payment_count = 0
poi_total_payment_count = 0

for key, value in enron_data.iteritems():
    if(value['salary'] <> 'NaN'):
        valid_salary_count = valid_salary_count + 1

    if (value['email_address'] <> 'NaN'):
        valid_email_count = valid_email_count + 1

    if (value['total_payments'] <> 'NaN'):
        total_payment_count = total_payment_count + 1
        if(value['poi'] == True):
            poi_total_payment_count = poi_total_payment_count + 1

print(valid_email_count)
print(valid_salary_count)
print(total_payment_count)
print(poi_total_payment_count)