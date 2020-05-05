# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:57:32 2020

@author: HP
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6,'abc':1,'def':2)

print(r.json())