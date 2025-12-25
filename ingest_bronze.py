

"""
This script ingests financial data from various sources, processes it, and loads it into Google BigQuery.

Features:
- Loads environment variables and configures logging
- Fetches data using vnstock and pandas
- Handles Google BigQuery operations
- Designed for use in the vstock data pipeline

Usage:
    python ingest_bronze.py

Environment:
- Requires a .env file for configuration
- Needs Google Cloud credentials for BigQuery access
"""

import os
import logging
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, GoogleAPIError
from vnstock import Listing, Quote
from datetime import datetime, date, timedelta, timezone
from dotenv import load_dotenv
from typing import List, Optional

# --- Configuration & Setup ---
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class BronzeConfig:
    """Configuration for the Bronze Layer."""
    PROJECT_ID = os.getenv("BQ_PROJECT_ID")
    DATASET_ID = os.getenv("BQ_DATASET_ID_BRONZE")
    TABLE_SYMBOLS = "stg_symbols"
    TABLE_PRICES = "stg_prices"
    
    # Schemas
    SCHEMA_SYMBOLS = [
        bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("is_vn100", "BOOLEAN"),
        bigquery.SchemaField("is_vn30", "BOOLEAN"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP"),
    ]
    
    SCHEMA_PRICES = [
        bigquery.SchemaField("time", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("open", "FLOAT"),
        bigquery.SchemaField("high", "FLOAT"),
        bigquery.SchemaField("low", "FLOAT"),
        bigquery.SchemaField("close", "FLOAT"),
        bigquery.SchemaField("volume", "INTEGER"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP"),
    ]

class BigQueryManager:
    """Handles BigQuery interactions."""
    def __init__(self):
        self.client = bigquery.Client(project=BronzeConfig.PROJECT_ID)
        self.dataset_ref = f"{BronzeConfig.PROJECT_ID}.{BronzeConfig.DATASET_ID}"

    def ensure_table_exists(self, table_name: str, schema: List[bigquery.SchemaField], partition_field: str = None, cluster_fields: List[str] = None):
        """Idempotent table creation."""
        table_id = f"{self.dataset_ref}.{table_name}"
        try:
            self.client.get_table(table_id)
            logger.info(f"Table exists: {table_name}")
        except NotFound:
            logger.info(f"Creating table: {table_name}")
            table = bigquery.Table(table_id, schema=schema)
            if partition_field:
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY, field=partition_field
                )
            if cluster_fields:
                table.clustering_fields = cluster_fields
            self.client.create_table(table)
            logger.info(f"Successfully created {table_name}")

    def upload_dataframe(self, df: pd.DataFrame, table_name: str, write_disposition: str = "WRITE_APPEND"):
        """Uploads a DataFrame to BigQuery."""
        if df.empty:
            logger.warning(f"No data to upload for {table_name}")
            return

        table_id = f"{self.dataset_ref}.{table_name}"
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            schema=BronzeConfig.SCHEMA_SYMBOLS if table_name == BronzeConfig.TABLE_SYMBOLS else BronzeConfig.SCHEMA_PRICES
        )
        
        try:
            job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
            job.result()  # Wait for completion
            logger.info(f"Uploaded {len(df)} rows to {table_name}")
        except GoogleAPIError as e:
            logger.error(f"Failed to upload to {table_name}: {e}")
            raise

    def query_symbols(self, filter_condition: str = "is_vn30 = TRUE") -> List[str]:
        """Fetches symbols from stg_symbols based on a condition."""
        query = f"""
            SELECT symbol 
            FROM `{self.dataset_ref}.{BronzeConfig.TABLE_SYMBOLS}` 
            WHERE {filter_condition}
        """
        try:
            results = self.client.query(query).result()
            return [row.symbol for row in results]
        except Exception as e:
            logger.error(f"Error querying symbols: {e}")
            return []

class BronzeIngestor:
    """Manages the ingestion logic."""
    def __init__(self):
        self.bq = BigQueryManager()

    def initialize_schema(self):
        """Ensures all necessary tables exist."""
        logger.info("Initializing Schema...")
        self.bq.ensure_table_exists(BronzeConfig.TABLE_SYMBOLS, BronzeConfig.SCHEMA_SYMBOLS)
        self.bq.ensure_table_exists(
            BronzeConfig.TABLE_PRICES, 
            BronzeConfig.SCHEMA_PRICES, 
            partition_field="time", 
            cluster_fields=["symbol"]
        )

    def ingest_symbol_master(self):
        """Refreshes the stg_symbols dimension table (Snapshot strategy)."""
        logger.info("Starting Symbol Ingestion...")
        try:
            vn100 = Listing().symbols_by_group('VN100')
            vn30 = Listing().symbols_by_group('VN30')
            
            df = pd.DataFrame(vn100, columns=['symbol'])
            df['is_vn100'] = True
            df['is_vn30'] = df['symbol'].isin(vn30)
            df['ingested_at'] = datetime.now(timezone.utc)
            
            # Use WRITE_TRUNCATE because this is a dimension snapshot
            self.bq.upload_dataframe(df, BronzeConfig.TABLE_SYMBOLS, write_disposition="WRITE_TRUNCATE")
        except Exception as e:
            logger.error(f"Failed to ingest symbols: {e}")

    def ingest_prices(self, symbol_list: List[str], lookback_days: int = 3):
        """
        Ingests price history.
        Strategy: Append-Only. 
        Note: Downstream Silver layer must handle deduplication (Qualify Row_Number).
        """
        start_date = (date.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = date.today().strftime('%Y-%m-%d')
        
        logger.info(f"Starting Price Ingestion for {len(symbol_list)} symbols. Range: {start_date} to {end_date}")
        
        batch_data = []
        BATCH_SIZE = 20  # Upload every 20 tickers to manage memory/network

        for i, symbol in enumerate(symbol_list):
            try:
                # Fetch Data
                df = Quote(symbol=symbol, source='VCI').history(start=start_date, end=end_date)
                
                if not df.empty:
                    # Transform
                    df['symbol'] = symbol
                    df['time'] = pd.to_datetime(df['time']).dt.date
                    df['ingested_at'] = datetime.now(timezone.utc)
                    
                    # Select only columns present in Schema to avoid errors
                    df = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'ingested_at']]
                    batch_data.append(df)
                
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")

            # Batch Upload
            if len(batch_data) >= BATCH_SIZE or (i == len(symbol_list) - 1 and batch_data):
                if batch_data:
                    logger.info(f"Uploading batch... (Symbols processed: {i+1}/{len(symbol_list)})")
                    combined_df = pd.concat(batch_data, ignore_index=True)
                    self.bq.upload_dataframe(combined_df, BronzeConfig.TABLE_PRICES, write_disposition="WRITE_APPEND")
                    batch_data = [] # Reset batch

# --- Main Execution ---
if __name__ == "__main__":
    ingestor = BronzeIngestor()
    
    # 1. Setup Database
    ingestor.initialize_schema()
    
    # 2. Update Symbol Master (Always refresh this first)
    ingestor.ingest_symbol_master()
    
    # 3. Define Strategy
    # Run daily with lookback=5 to catch weekends/holidays safely.
    # Set lookback=5000 for the initial backfill run.
    IS_BACKFILL = False
    if IS_BACKFILL:
        LOOKBACK = 3000
    elif date.today().weekday() == 0:  # Monday
        LOOKBACK = 3
    else:
        LOOKBACK = 1

    # 4. Fetch Target Symbols
    # We query the BQ table we just updated to ensure consistency
    target_symbols = ingestor.bq.query_symbols(filter_condition="is_vn100 = TRUE")
    
    # 5. Execute Price Ingestion
    if target_symbols:
        ingestor.ingest_prices(target_symbols, lookback_days=LOOKBACK)
    else:
        logger.warning("No symbols found to process.")