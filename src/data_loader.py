from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
CONFIG_FILES_DIR = BASE_DIR / 'config_files'
MODELS_DIR = BASE_DIR / 'models'
IPC_PATH = BASE_DIR / 'IPC'

def csv_to_parquet(DIR, filename, filename_parquet, rewrite = False, chunksize = 100000):
    csv_stream = pd.read_csv(DIR / filename, chunksize = chunksize)
    parquet_file = PROCESSED_DATA_DIR / filename_parquet
    if (PROCESSED_DATA_DIR / filename_parquet).exists() and rewrite is True:
        PROCESSED_DATA_DIR / filename_parquet.unlink()

    elif (PROCESSED_DATA_DIR / filename_parquet).exists():
        print('File already exists')

    for i, chunk in enumerate(csv_stream):
        if i == 0:
            parquet_schema = pa.Table.from_pandas(chunk).schema
            parquet_writer = pq.ParquetWriter(parquet_file, schema=parquet_schema, compression='snappy')
        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)
    parquet_writer.close()

