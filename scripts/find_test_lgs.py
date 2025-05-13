from pathlib import Path
import csv
import geopy.distance
import pandas as pd
import numpy as np
from collections import OrderedDict

# Define paths
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data" 


file_path = DATA_PATH / "grambank.encoded.tsv"
w_file_path = "C:/Users/theod/Desktop/code/grambank/data/grambank.condensed.tsv"
all_rows = [] #list of dictionaries for each row 
with open(file_path, encoding="utf-8") as f:
    d2 = csv.DictReader(f, delimiter="\t")   #read tsv, not csv 
    for r in d2:
        all_rows.append(r)
#print(all_rows[0])
#{'id': 'abad1241', 'language_name': 'Abadi', 'language_latitude': '-9.03389', 'language_longitude': '146.992', 'language_macroarea': 'Papunesia', 'language_family': 'Austronesian', 'GB020_ARTDef': '', 'GB021_ARTIndef': '', 'GB022_ARTPre': '', 'GB023_ARTPost': '', 'GB026_ADJDiscont': 'False',
def all_areas(all_rows):
    areas= set()
    for i in all_rows:
        area=i['language_macroarea']
        areas.add(area)
    return areas 

#print(all_areas(all_rows))
#{'', 'Papunesia', 'Africa', 'South America', 'North America', 'Australia', 'Eurasia'}
areas= ['Papunesia', 'Australia', 'Eurasia' ]
def find_lgs(all_rows, areas):
    family_c={}
    selected_families= []
    for i in all_rows:
        if i['language_macroarea'] in areas: # if a lg belongs to that area
            fam= i['language_family'] #the fam it belongs to 
            if fam in family_c:
                family_c[fam] +=1
            else:
                family_c[fam] =1

    for family, count in family_c.items():
        if 8<count<16:                        #range adjusted so all areas can be found 
            selected_families.append(family)

    return selected_families   
                

print(find_lgs(all_rows, areas))
            


    
