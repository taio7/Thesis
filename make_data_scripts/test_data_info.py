#chat
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


def count_lgs(all_rows, family):
    c=0
    area= None 
    for l in all_rows:
        if l["language_family"]== family:
            c+= 1
            area= l["language_macroarea"]
    print(f"the family {family} has {c} langugaes and the area {area}")
    return c, area 

families=["Mayan", "Tucanoan", "Madang", "Mahakiranti", "Northern Pama-Nyungan", "Nilotic"]
for i in families:
    count_lgs(all_rows, i)
#missing new guinea , nepal india, australia 
#if m["genus"] == "Mayan" or m["genus"] == "Tucanoan" or m["genus"] == "Madang" or m["genus"] == "Mahakiranti" or m["genus"] == "Northern Pama-Nyungan" or m["genus"] == "Nilotic":

print(all_rows[:2])