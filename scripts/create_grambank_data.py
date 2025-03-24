"""
Utility script to build the databases that can be fed into the models for analyses.

This script is intentionally a pure Python script, and does not use any of the
CLDF libraries for manipulation in order to keep the dependencies to a minimum.
"""

# Import Python standard libraries
from pathlib import Path
from typing import Dict, List
import csv
import logging
import zipfile

# Import 3rd-party libraries
import pandas as pd
import numpy as np

# Define the paths
BASE_PATH = Path(__file__).parent
#unzip grambank file 
with zipfile.ZipFile("grambank-v1.0.3.zip", "r") as zip_ref:
    zip_ref.extractall(BASE_PATH)
#change the path, this is where the data is
GRAMBANK_PATH = BASE_PATH / "grambank-grambank-7ae000c" / "cldf"


def load_grambank_languages() -> Dict[str, Dict]:
    """
    Load the GramBank language data.

    Returns:
        A dictionary of language data, keyed by the language ID.
    """

    # Load the language data from `languages.csv` as a dictionary,
    # using the `ID` column as the key and the rest of the columns
    # as the values
    languages = {}
    with open(GRAMBANK_PATH / "languages.csv", encoding="utf-8") as languages_file:
        reader = csv.DictReader(languages_file)
        for row in reader:
            row_id = row.pop("ID")
            latitude = row.pop("Latitude")
            longitude = row.pop("Longitude")
            if latitude and longitude:
                latitude = float(latitude)
                longitude = float(longitude)

            languages[row_id] = {
                "name": row.pop("Name"),
                "latitude": latitude,
                "longitude": longitude,
                "macroarea": row.pop("Macroarea"),
                "family": row.pop("Family_name"),
                "family_id": row.pop("Family_level_ID"),
                "glottocode": row.pop("Glottocode"),
                "language_id": row.pop("Language_level_ID"),
                "level": row.pop("level"),
                "lineage": row.pop("lineage").split("/"),
            }

    return languages


def load_grambank_parameters() -> Dict[str, Dict]:
    """
    Load the GramBank parameter data.

    Returns:
        A dictionary of parameter data, keyed by the parameter ID.
    """

    # Load the parameter data from `parameters.csv` as a dictionary,
    # using the `ID` column as the key and the rest of the columns
    # as the values
    parameters = {}
    with open(GRAMBANK_PATH / "parameters.csv", encoding="utf-8") as parameters_file:
        reader = csv.DictReader(parameters_file)
        for row in reader:
            row_id = row.pop("ID")
            parameters[row_id] = {
                "name": row.pop("Name"),
                "description": row.pop("Grambank_ID_desc").replace(" ", "_"),
            }

    return parameters


def load_grambank_values() -> List[Dict]:
    """
    Load the GramBank values data.

    Returns:
        A list of value data.
    """

    # Load the values data from `values.csv` as a dictionary,
    # using the `ID` column as the key and the rest of the columns
    # as the values
    values = []
    with open(GRAMBANK_PATH / "values.csv", encoding="utf-8") as values_file:
        reader = csv.DictReader(values_file)
        for row in reader:
            values.append(
                {
                    "id": row["ID"],
                    "language_id": row["Language_ID"],
                    "parameter_id": row["Parameter_ID"],
                    "value": int(row["Value"]) if row["Value"] != "?" else None,
                    "code_id": row["Code_ID"],
                }
            )

    return values


def load_grambank_data() -> List[Dict]:
    """
    Load the GramBank data, parsing it into a single table.

    Returns:
        A list of value data.
    """

    # Load the language data
    logging.info("Loading language data")
    languages = load_grambank_languages()

    # Load the parameter data
    logging.info("Loading parameter data")
    parameters = load_grambank_parameters()

    # Load the values data
    logging.info("Loading values data")
    values = load_grambank_values()

    # Iterate over the values and build the single table
    logging.info("Building single data table")
    entries = []
    for entry in values:
        language_id = entry["language_id"]
        parameter_id = entry["parameter_id"]

        language_family = languages[language_id]["family"]
        if not language_family:
            language_family = languages[language_id]["name"]
        entries.append(
            {
                "id": entry["id"],
                "language_id": language_id,
                "language_name": languages[language_id]["name"],
                "language_latitude": languages[language_id]["latitude"],
                "language_longitude": languages[language_id]["longitude"],
                "language_macroarea": languages[language_id]["macroarea"],
                "language_family": language_family,
                "parameter_id": parameters[parameter_id]["description"],
                "value": entry["value"],
            }
        )

    # Sort entries by "parameter_id" and "language_id"
    entries = sorted(entries, key=lambda x: (x["parameter_id"], x["language_id"]))

    return entries


def build_long_table_df(grambank_data: List[Dict]):
    """
    Build the long table data frame.

    :parameter grambank_data: The GramBank data.
    """

    # Write the data to disk as a single TSV file
    #create data dir first 
    (BASE_PATH / "data").mkdir(parents=True, exist_ok=True)
    logging.info("Writing longtable data to disk (%i entries)", len(grambank_data))
    with open(
        BASE_PATH / "data" / "grambank.longtable.tsv", "w", newline="", encoding="utf-8"
    ) as grambank_file:
        writer = csv.DictWriter(
            grambank_file,
            fieldnames=[
                "id",
                "language_id",
                "language_name",
                "language_latitude",
                "language_longitude",
                "language_macroarea",
                "language_family",
                "parameter_id",
                "value",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(grambank_data)


def build_condensed_table_df(grambank_data: List[Dict]):
    """
    Build the condensed table data frame.

    :parameter grambank_data: The GramBank data.
    """

    # Collect a list of all parameter IDs
    parameter_ids = set()
    for entry in grambank_data:
        parameter_ids.add(entry["parameter_id"])
    parameter_ids = sorted(parameter_ids)

    # Collect a dictionary with all metalinguistic information (columns "language_name",
    # "language_latitude", "language_longitude", "language_macroarea", "language_family"),
    # keyed by the language ID
    language_info = {}
    for entry in grambank_data:
        language_id = entry["language_id"]
        if language_id not in language_info:  #add this only once in dict
            language_info[language_id] = {
                "language_name": entry["language_name"],
                "language_latitude": entry["language_latitude"],
                "language_longitude": entry["language_longitude"],
                "language_macroarea": entry["language_macroarea"],
                "language_family": entry["language_family"],
            }

    # Extend the "language_info" dictionary by adding a column for each parameter
    # (all here initialized to `None`)
    for language_id in language_info:
        for parameter_id in parameter_ids:
            language_info[language_id][parameter_id] = None

    # Iterate over the GramBank data and fill in the values
    for entry in grambank_data:
        language_id = entry["language_id"]
        parameter_id = entry["parameter_id"]
        language_info[language_id][parameter_id] = entry["value"]

    # Build a list of dictionaries from "language_info", adding a key "id" with the
    # language ID as the value
    condensed_data = []
    for language_id in language_info:
        entry = language_info[language_id]
        entry["id"] = language_id
        condensed_data.append(entry)

    # Write the data to disk as a single TSV file
    logging.info("Writing condensed data to disk (%i entries)", len(condensed_data))
    with open(
        BASE_PATH / "data" / "grambank.condensed.tsv", "w", newline="", encoding="utf-8"
    ) as grambank_file:
        writer = csv.DictWriter(
            grambank_file,
            fieldnames=[
                "id",
                "language_name",
                "language_latitude",
                "language_longitude",
                "language_macroarea",
                "language_family",
            ]
            + parameter_ids,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(condensed_data)


def build_encoded_df():
    """
    Build the encoded data frame.

    We load the data directly from the TSV file, and then encode the
    categorical data with pandas.
    """

    # Load data
    file_path = BASE_PATH / "data" / "grambank.condensed.tsv"
    df = pd.read_csv(file_path, sep="\t")

    # Obtain the feature columns
    gb_columns = [col for col in df.columns if col.startswith("GB")]

    # Mark feature columns as integers (dealing with NAs)
    for col in gb_columns:
        df[col] = df[col].astype(pd.Int64Dtype())

    # Filter 'gb_columns' with more than two distinct values (not counting NAs)
    gb_columns_more_than_two = [
        col for col in gb_columns if df[col].nunique(dropna=True) > 2
    ]

    # Create an empty DataFrame to store the one-hot encoded columns
    df_encoded = pd.DataFrame()

    # Iterate through the columns to be one-hot encoded
    for column in gb_columns_more_than_two:
        # Perform one-hot encoding on the current column
        encoded = pd.get_dummies(df[column], prefix=column, dummy_na=False)

        # Set all one-hot encoded columns to np.nan where the original value was NA
        encoded[df[column].isna()] = np.nan

        # Concatenate the encoded columns to the new DataFrame
        df_encoded = pd.concat([df_encoded, encoded], axis=1)

    # Drop 'gb_columns_more_than_two' from the original DataFrame 'df'
    df = df.drop(columns=gb_columns_more_than_two)

    # Calculate the difference between 'gb_columns' and 'gb_columns_more_than_two', and
    # convert columns with only two distinct values to boolean type while handling NaN values
    gb_columns_two_values = list(set(gb_columns) - set(gb_columns_more_than_two))
    for col in gb_columns_two_values:
        df[col] = df[col].astype(pd.BooleanDtype()) 

    # Join 'df_encoded' to 'df'
    df = pd.concat([df, df_encoded], axis=1)

    # Write the data to disk as a single TSV file
    logging.info("Writing encoded data to disk (%i entries)", len(df))
    df.to_csv(BASE_PATH / "data" / "grambank.encoded.tsv", sep="\t", index=False)


def main():
    """
    Main entry point for the script.
    """

    # Load the grambank data as a single table
    grambank_data = load_grambank_data()

    # Build the long table data
    build_long_table_df(grambank_data)

    # Build the condensed table data
    build_condensed_table_df(grambank_data)

    # Build the encoded data frame
    build_encoded_df()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()