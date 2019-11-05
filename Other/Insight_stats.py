#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:04:26 2019

@author: antoinepetit
"""

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
import re
import itertools

import matplotlib.pyplot as plt


# SCRAPE AND FORMAT LIST OF INSIGHT DATA FELLOWS
# Request web page
for pp in range(9):
    if pp == 0:
        page = requests.get('https://www.insightdatascience.com/fellows')
    else:
        page = requests.get('https://www.insightdatascience.com/fellows?61ea5d1b_page='+str(pp))
    
    # Parse HTML and get Tag element
    soup = BeautifulSoup(page.content, 'html.parser')
    html = list(soup.children)[2]
    
    # Define classes of interest
    class_names = {'name': 'tooltip_name',
                   'title': 'toottip_title',
                   'company': 'tooltip_company',
                   'insight_project': 'tooltip_project',
                   'background': 'tooltip_background'}
    
    # Make dataframe of fellow information
    class_htmls = {k:soup.find_all('div', class_=v) for (k,v) in class_names.items()}
    if pp==0:
        df_fellows = pd.DataFrame({k: [class_htmls[k][i].get_text() for i in range(len(class_htmls[k]))] for k in class_names.keys()})
    else:
        df_fellows = pd.concat([df_fellows, pd.DataFrame({k: [class_htmls[k][i].get_text() for i in range(len(class_htmls[k]))] for k in class_names.keys()})]).reset_index(drop=True)

# Parse academic background
# NOTE: parsing of academic information is not possible in some cases because of inconsistencies in string formatting
df_fellows['N_commas'] = [s.count(',') for s in df_fellows['background']]
df_fellows['academic_field'] = np.nan
df_fellows['academic_institute'] = np.nan
df_fellows['academic_position'] = np.nan
df_fellows.loc[df_fellows['N_commas']==2, 'academic_field'] = [x.split(',')[0].strip() for x in df_fellows.loc[df_fellows['N_commas']==2, 'background']]
df_fellows.loc[df_fellows['N_commas']==2, 'academic_institute'] = [x.split(',')[1].strip() for x in df_fellows.loc[df_fellows['N_commas']==2, 'background']]
df_fellows.loc[df_fellows['N_commas']==2, 'academic_position'] = [x.split(',')[2].strip() for x in df_fellows.loc[df_fellows['N_commas']==2, 'background']]

# List of affiliations that could not be parsed (uncommon comma structure)
backgrounds_unparsed = df_fellows.loc[df_fellows['N_commas']!=2]['background'].values

## Misc formatting
#df_fellows.drop('N_commas', axis=1, inplace=True)
#df_fellows['title'] = [x.strip() for x in df_fellows['title']]
#
# Rename some institutes that are duplicates that occur more than once
institute_renames = {'Stanford University': 'Stanford',
                     'Harvard University': 'Harvard',
                     'Harvard-Smithsonian Center for Astrophysics': 'Harvard',
                     'Harvard Medical School': 'Harvard',
                     'Columbia University': 'Columbia',
                     'Princeton University': 'Princeton',
                     'Yale University': 'Yale',
                     'New York University': 'NYU',
                     'UCSD': 'UC San Diego',
                     'University of California - San Diego': 'UC San Diego',
                     'UCSF': 'UC San Francisco',
                     'UCSB': 'UC Santa Barbara',
                     'UCSC': 'UC Santa Cruz',
                     'University of Texas at Austin': 'UT Austin',
                     'Massachusetts Institute of Technology': 'MIT',
                     'California Institute of Technology': 'Caltech',
                     'Georgia Institute of Technology': 'Georgia Tech'}
df_fellows['academic_institute'] = df_fellows['academic_institute'].replace(institute_renames)

df_fellows['location'] = 0
loc_dic = {"NYC":["Birchbox","New York Times", "Bloomberg", "Wall Street Journal",
"Spotify","Palantir","Capital One","AT&T","Verizon","NBC","Oscar",
"MTV","Wink","Foursquare","Meetup","Jet","Viacom","ZocDoc","Blackrock","Morgan",
"enigma","okcupid","Descartes","McKinsey","Viacom","Atlassian","OnDeck",
"AlixPartners","IBM","Macy","Uber","Accenture","Goldman Sachs","Schireson"
"AbleTo","Arena","Blue Apron","Dataiku","DigitalOcean","Dstillery","EnergyHub",
"Fareportal","FINRA","Flatiron Health","Hudson River Trading","JW Player",
"Komodo Health","Markable","Murmuration","Neo Ivy Capital Management","Seatgeek",
"Showtime","Simulmedia","Splash","Statespace","Tapad","Thasos Group","Via",
"Viacom"],
"WA":["Convoy","SAP", "Concur","Microsoft","Facebook","Amazon","AirBnB","KPMG",
"Uber","Zillow","Indeed","iSpot","Zymergen","RealSelf","Lyft",
"ProCogia","Boston Consulting Group","Expedia","Liberty Mutual","Schireson",
"Bsquare"],
"SF":["Square","Facebook","LinkedIn","Twitter","Airbnb","Uber","Microsoft","Apple",
"Netflix","Stitch Fix","Reddit","Fitbit","Twitch","Intuit","Pinterest","Etsy","Premise",
"LiveRamp","Hipmunk","Stealth Security","SambaTV","Paypal","Swish","Adobe",
"Youtube","Google","Accenture","Cisco","eBay","GoPro","H20","Pandora","Target",
"Walmart Labs","Yelp","Bosch","23","6Sense","AdRoll","Affirm","App Annie",
"Bosch","Cainthus","Cape Analytics","Counsyl","Domino Data Lab","Doxmity",
"Enlitic","Eventbrite","Headspin","InVitae","Juvo","LendUp","Livongo","Lumiata",
"NerdWallet","Omniscience","Opendoor","Optimizely","Proofpoint","Prosper",
"Quora","RadiumOne","Rhumbix","RichRelevance","Robinhood","Symantec","Synopsys",
"TaskRabbit","Electronic Arts","Salesforce"],
'OTHER':["Allstate","Citadel","Credit Suisse","CVS","Gartner","GEICO","Goodyear",
"HP","Intel","Jawbone","Lululemon","Mercedes","Nestl","Nielsen","Oracle",
"Pearson","Sonos","Tesla","TripAdvisor","3M","Activision","Akamai","ARM",
"Backflip Studios","Beyond Limits","BitSight","Booster","Caesars Entertainment Corporation",
"Calabrio","Catalina","CiBO","Cinch","Colaberry","Compass","Cotiviti","CubeSmart",
"DataXu","Guggenhiem Partners","HP","Hubspot","Illumina","Indigo","Intertek","IQVIA",
"Kabbage","LA Dodgers","Marlette Funding","McKesson","Nauto","Reilly",
"Osram","Enthought","Pebble","Proteus","Rubicon Project","Sentry Data Systems",
"Seven Bridges","State Farm","Tamr","The Honest Company","Vanguard","Vectra",
"Vroom","Wellington","ZestFinance","Vizient","Vistex"]}

loc_list = []
for idx,item in enumerate(df_fellows.iterrows()):
    test = item[1]['company']
    none_found = True
    for ii in range(len(loc_dic)):
        if any(name in test.lower() for name in map(lambda x:x.lower(),loc_dic[loc_dic.keys()[ii]])):
            loc_list.append(loc_dic.keys()[ii])
            none_found = False
            break
    if none_found:
        loc_list.append('NaN')

new_df = pd.DataFrame({'location':loc_list})
df_fellows.update(new_df)
print "NaN: " + str(df_fellows[df_fellows.location == 'NaN'].shape[0])
print "SF: " + str(df_fellows[df_fellows.location == 'SF'].shape[0])
print "NYC: " + str(df_fellows[df_fellows.location == 'NYC'].shape[0])
print "WA: " + str(df_fellows[df_fellows.location == 'WA'].shape[0])
print "OTHER: " + str(df_fellows[df_fellows.location == 'OTHER'].shape[0])
