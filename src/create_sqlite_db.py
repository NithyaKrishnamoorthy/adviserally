"""
Downloads excel files from S3, creates sqlite database from excel file and uploads to S3.

"""
import os
import sqlite3
import sys
from typing import List

import boto3
import pandas as pd
from loguru import logger


def download_raw_excel_files(excel_files: List[str] = ["scrapped_output_term_full_20231011.xlsx"]) -> None:
    """Download excel files from S3."""
    # Initialize the S3 client
    s3 = boto3.client("s3")

    # download multiple excel files from s3
    for file in excel_files:
        # Define the bucket name and the file key
        bucket_name = "for-singlife"
        file_key = f"projects/advisorally/{file}"
        download_path = f"data/excel/{file}"

        # Download the file
        s3.download_file(bucket_name, file_key, download_path)
        logger.info(f"Downloaded {file_key} from {bucket_name} to {download_path}")


def upload_sqlite_to_s3(database_name) -> None:
    """Upload SQLite database to S3."""
    # Initialize the S3 client
    s3 = boto3.client("s3")

    # Define the bucket name and the file key
    bucket_name = "for-singlife"
    file_key = f"projects/advisorally/{database_name}"
    upload_path = f"data/sqlite/{database_name}"

    # Upload the file
    s3.upload_file(upload_path, bucket_name, file_key)
    logger.info(f"Uploaded {upload_path} to {bucket_name} as {file_key}")


if __name__ == "__main__":
    excel_files = ["scrapped_output_term_full_20231011.xlsx"]
    database_name = "my_database.sqlite"

    # check if the database exists
    if os.path.exists(f"data/sqlite/{database_name}"):
        logger.info(
            f"SQLite database already exists in data/sqlite/{database_name}, stopping creation of sqlite database"
        )
        sys.exit(0)

    logger.info("SQLITE database does not exist, creating sqlite database")

    download_raw_excel_files(excel_files)
    # Read the Excel file into a DataFrame.
    # TODO: will need to change this block depending on the excel file
    df = pd.read_excel("data/excel/scrapped_output_term_full_20231011.xlsx", sheet_name="Term Life", index_col=0)

    # Create a connection to the SQLite database (it will create a new file if it doesn't exist)
    conn = sqlite3.connect(f"data/sqlite/{database_name}")

    # Write the DataFrame to the SQLite database
    df.to_sql("term_life_table", conn, if_exists="replace", index=True)
    logger.info(f"Successfully created SQLite database in data/sqlite/{database_name}")
    # Close the connection
    conn.close()

    upload_sqlite_to_s3(database_name)
