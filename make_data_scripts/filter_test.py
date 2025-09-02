import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"

INPUT_TEST_FILE = DATA_PATH/"test.tsv"
OUTPUT_FILTERED_FILE = DATA_PATH/"test_gold_rand.tsv"

included_fams = {"Mayan", "Tucanoan", "Sepik", "Worrorran", "Tungusic", "Nilotic"}
num_test_lgs= 100


df = pd.read_csv(INPUT_TEST_FILE, sep="\t", dtype=str)

#keep all lgs from the families 
from_fams = df[df["language_family"].isin(included_fams)].copy()

rest= (num_test_lgs)-(len(from_fams))
#print(rest) #38
not_included=[]
for i, row in df.iterrows():
    if row["language_family"] not in included_fams:
        not_included.append(row)

not_included=pd.DataFrame(not_included)

sample= not_included.sample(n=rest, random_state=42)
final= pd.concat([sample, from_fams], ignore_index=True) 

final.to_csv(OUTPUT_FILTERED_FILE, sep="\t", index=False)

print("done")