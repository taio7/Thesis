# Python 3.7.6
#Feature column format as in sigtyp 
from pathlib import Path
import csv
import geopy.distance
from collections import OrderedDict
import pandas as pd
import numpy as np

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data" 


file_path = DATA_PATH / "grambank.encoded.tsv"

all_rows = [] #list of dictionaries for each row 
with open(file_path, encoding="utf-8") as f:
    d2 = csv.DictReader(f, delimiter="\t")   #read tsv, not csv 
    for r in d2:
        all_rows.append(r)
     

#print(all_rows[:5])
#print(f"before filter {len(all_rows)}") == 2467

# select those languages having more than 3 features
all4 = []
for l in all_rows:
    new_row = {}       #chnage to 6 for GBdata, includes only features 
    f = list(l.items())[6:] #list of tuples= [(id,aba124),(lgname, Abadi)...]
    #print(f)
    c = 0
    ff = []
    for t  in f:
        if t[1] != "":  #if the valus of the feature isnt empty
            c += 1
            ff.append(t)
    if c > 50:
          m = list(l.items())[0:6] 
          for b in m:                 #store metadata in new rwo
              new_row[b[0]] = b[1]
          for h in ff:                 #store non-empty features to new row
              new_row[h[0]] = h[1]
          all4.append(new_row)
#print(all4[0])
#print(f"after filter {len(all4)}") == 2424                            
# find geo-coordinates for some languages
#print(m)
otherlang_coo = []
excluded_lgs= ["Mayan", "Tucanoan", 'Sepik', 'Worrorran', 'Tungusic', 'Nilotic' ]
for m in all4:
    if m['language_family'] in excluded_lgs:
      cc = (m["language_latitude"], m["language_longitude"])
      otherlang_coo.append(cc)
#print(len(otherlang_coo))  == 77 = test languages 
# find languages which do not belong to certain genera and are distant more than 1000 km
pre_fin = [] 
for m in all4: 
      coord1 = (m["language_latitude"], m["language_longitude"])
      distances = []
      for coord2 in otherlang_coo: 
         dist = geopy.distance.geodesic(coord1, coord2)
         distances.append(dist)
      if m['language_family'] not in excluded_lgs and  all(i > 1000 for i in distances): 
         w = m['id'] 
         fa = m['language_family']  
         f = list(m.items())[6:] 
         pre_fin.append([w, fa, f]) 
#print(len(pre_fin)) =1607 = train languages left 
#['abad1241', 'Austronesian', [('GB026_ADJDiscont', 'False'), ('GB027_ComitConjDifferent', 'False')
#  find the set of all features of the above languages, not in fams and not within 1000 km from them
all_feat = set()
for a,b,c in pre_fin: 
    for k in c: 
         all_feat.add(k[0]) 


# extract from all_feat above the features that are in more than 9 languages
fts = {}
for f in all_feat:  #f= feature code 
   count = 0
   for a,b,c in pre_fin:  #[id, fam, [(feature, value)]]
      for d,e in c:  #feature , value 
          if f == d:  #if feture code form all features with values is in filtered lgs
              count += 1
   if count > 9: #no features excluded,, with >1000 fetures from 209 to 207
     fts[f] = count
#print(len(fts))= 209 features excluded due to being too rare 


a = list(fts.keys()) #['GB188_AUGBound', 'GB027_ComitConjDifferent']

fin = []
for m in all4:
    coord1 = (m["language_latitude"], m["language_longitude"])
    distances = []
    for coord2 in otherlang_coo:
       dist = geopy.distance.geodesic(coord1, coord2)
       distances.append(dist)
    if m['language_family'] not in excluded_lgs and  all(i > 1000 for i in distances): 
        #GBformat 
        row= [
            m['id'],
            m["language_name"],
            m["language_latitude"],
            m["language_longitude"],
            m['language_macroarea'],
            m["language_family"]]
        f = dict(list(m.items())[6:])
        for feat in a:
            row+= [f.get(feat,"" )]
        fin.append(row)
#print(fin[:2])
    """#sigtyp format 
        w = m['id']
        name = m["language_name"]
        lat = m["language_latitude"]
        lon = m["language_longitude"]
        ma = m['language_macroarea'] 
        gen = m["language_family"]  #replace with existing comlumns only
        f = list(m.items())[6:]
        nn = []
        for n in f:
            if n[0] in a:
            # j = n[0].split(" ", 1) no need to reformat, already this format for gbdata
                s = n[0]+ "=" + n[1]
                nn.append(s)
        if len(nn) > 3:  # to be sure the features are at least 4
           vv = "|".join(nn)
           fin.append(w + "\t" + name + "\t" + lat + "\t" + lon + "\t" + ma + "\t" + gen + "\t" + vv)
#print(fin[0])"""
new_fin = fin[0: round(len(fin) * 0.9)]
sel10 = fin[round(len(fin) * 0.9):]
dev_add = sel10[:round(len(sel10) * 0.5)]
test_add = sel10[round(len(sel10) * 0.5):]
header = [
    "id", "language_name", "language_latitude", "language_longitude",
    "language_macroarea", "language_family"
] + a
# save train data
with open(DATA_PATH/"train.tsv", "wt", encoding="utf-8", newline="") as f:
    writer=csv.writer(f, delimiter="\t")
    writer.writerow(header)
    writer.writerows(new_fin)


# crete dev and test
all_ex = []
for l in all_rows:
    if l['language_family'] in excluded_lgs:
    #if l["genus"] == "Mayan" or l["genus"] == "Tucanoan" or l["genus"] == "Madang" or l["genus"] == "Mahakiranti" or l["genus"] == "Northern Pama-Nyungan" or l["genus"] == "Nilotic":
        new_row = OrderedDict()
        f = list(l.items())[6:]
        c = 0
        ff = []
        for t  in f:
            if t[1] != "":
                c += 1
                ff.append(t)
        
        if c > 3:
            row = [
                l['id'],
                l["language_name"],
                l["language_latitude"],
                l["language_longitude"],
                l['language_macroarea'],
                l["language_family"]
            ]
            f_dict= dict(ff)
            for feat in a:
                row += [f_dict.get(feat, "")]
            all_ex.append(row)
            """w = l['id']
            name = l["language_name"]
            lat = l["language_latitude"]
            lon = l["language_longitude"]
            ma = l['language_macroarea'] 
            fa = l["language_family"]  #replace with existing comlumns only
            nn = []
            for n in ff:
                if n[0] in a:
                  #j = n[0].split(" ", 1)
                  s = n[0]+ "=" + n[1]
                  nn.append(s)
            if len(nn) > 3:  # to be sure the features are at least 4
              vv = "|".join(nn)
              all_ex.append(w + "\t" + name + "\t" + lat + "\t" + lon + "\t" + ma + "\t" + fa + "\t" + vv)"""

# select 20%
p20 = round(len(all_ex) * 0.2) 

dev = all_ex[0: p20] + dev_add

# save dev data
with open(DATA_PATH/"dev.tsv", "wt", encoding="utf-8", newline='') as f:
    writer=csv.writer(f, delimiter="\t")
    writer.writerow(header)
    writer.writerows(dev)
