"""
This script ingests financial data from vnstock, processes it, and loads it into Google BigQuery.

Features:
- Bronze Layer Ingestion (Raw Data)
- "Time-Chunking" strategy for massive backfills (avoids 4000 partition limit)
- "Ticker-Batch" strategy for fast daily updates
- Idempotent table creation
- Comprehensive logging and error handling

Usage:
    python ingest_bronze.py
"""

import os
import logging
import time
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, GoogleAPIError
from vnstock import Listing, Quote
from datetime import datetime, date, timedelta, timezone
from dotenv import load_dotenv
from typing import List, Tuple

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
    # Load from .env or use defaults
    PROJECT_ID = os.getenv("BQ_PROJECT_ID")
    DATASET_ID = os.getenv("BQ_DATASET_ID_BRONZE") 
    
    TABLE_SYMBOLS = "stg_symbols"
    TABLE_PRICES = "stg_prices_temp"
    
    # Schema Definition: Symbol Master
    SCHEMA_SYMBOLS = [
        bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("is_vn100", "BOOLEAN"),
        bigquery.SchemaField("is_vn30", "BOOLEAN"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP"),
    ]
    
    # Schema Definition: Stock Prices
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
    """Handles low-level BigQuery interactions."""
    def __init__(self):
        self.client = bigquery.Client(project=BronzeConfig.PROJECT_ID)
        self.dataset_ref = f"{BronzeConfig.PROJECT_ID}.{BronzeConfig.DATASET_ID}"

    def ensure_table_exists(self, table_name: str, schema: List[bigquery.SchemaField], partition_field: str = None, cluster_fields: List[str] = None):
        """Checks if a table exists; if not, creates it with partitioning/clustering."""
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
        """Uploads a pandas DataFrame to BigQuery."""
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
            job.result()  # Wait for job completion
            logger.info(f"Uploaded {len(df)} rows to {table_name}")
        except GoogleAPIError as e:
            logger.error(f"Failed to upload to {table_name}: {e}")
            raise

    def query_symbols(self, filter_condition: str = "is_vn30 = TRUE") -> List[str]:
        """Fetches a list of symbols from the stg_symbols table."""
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
    """Manages the high-level ingestion logic."""
    def __init__(self):
        self.bq = BigQueryManager()

    def initialize_schema(self):
        """Ensures all necessary tables are ready."""
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

    # --- DAILY RUN STRATEGY ---
    def ingest_prices_daily(self, symbol_list: List[str], lookback_days: int = 3):
        """
        Lightweight method for daily updates.
        Strategy: Loop Tickers -> Batch Upload.
        Best for small date ranges (1-5 days).
        """
        start_date = (date.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = date.today().strftime('%Y-%m-%d')
        
        logger.info(f"--- Daily Ingestion: {start_date} to {end_date} ---")
        
        batch_data = []
        BATCH_SIZE = 50 

        for i, symbol in enumerate(symbol_list):
            try:
                # Fetch Data
                df = Quote(symbol=symbol, source='VCI').history(start=start_date, end=end_date)
                
                if not df.empty:
                    # Transform
                    df['symbol'] = symbol
                    df['time'] = pd.to_datetime(df['time']).dt.date
                    df['ingested_at'] = datetime.now(timezone.utc)
                    df = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'ingested_at']]
                    batch_data.append(df)
            
            except Exception:
                # Silently skip errors in daily run to keep logs clean, or use logger.debug()
                pass 

            # Upload when batch is full or at end of list
            if len(batch_data) >= BATCH_SIZE or (i == len(symbol_list) - 1):
                if batch_data:
                    logger.info(f"Uploading batch... (Processed {i+1}/{len(symbol_list)} symbols)")
                    self.bq.upload_dataframe(pd.concat(batch_data), BronzeConfig.TABLE_PRICES)
                    batch_data = [] # Reset batch

    # --- BACKFILL STRATEGY (QUOTA SAFE) ---
    def backfill_history(self, symbol_list: List[str], start_str: str, end_str: str):
        logger.info(f"🚀 Starting Smart Backfill: {start_str} to {end_str}")
        
        all_data = []
        total_symbols = len(symbol_list)
        request_counter = 0 # <--- 1. Khởi tạo biến đếm
        
        # BƯỚC 1: Tải dữ liệu
        for i, symbol in enumerate(symbol_list):
            request_counter += 1 # <--- 2. Tăng đếm lên 1
            
            try:
                # In ra số thứ tự request ngay lập tức
                print(f"📡 Request [{request_counter}/{total_symbols}] -> Sending to VCI for {symbol}...")

                # Ngủ nhẹ để tránh Rate Limit
                time.sleep(0.5) 
                
                # Gửi request thực sự
                df = Quote(symbol=symbol, source='VCI').history(start=start_str, end=end_str)
                
                if df is not None and not df.empty:
                    df['symbol'] = symbol
                    df['time'] = pd.to_datetime(df['time']).dt.date
                    df['ingested_at'] = datetime.now(timezone.utc)
                    df = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'ingested_at']]
                    all_data.append(df)
                    print(f"   ✅ OK: {len(df)} rows fetched.")
                else:
                    print(f"   ⚠️ Empty response.")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")

        if not all_data:
            logger.warning("No data fetched.")
            return

        # BƯỚC 2: Gộp dữ liệu
        logger.info(f"Merging {len(all_data)} dataframes in memory...")
        full_df = pd.concat(all_data, ignore_index=True)
        
        # BƯỚC 3: Cắt nhỏ và Upload
        split_date = date(2018, 1, 1)
        
        df_part1 = full_df[full_df['time'] < split_date]
        if not df_part1.empty:
            logger.info(f"📤 Uploading Part 1 (< 2018): {len(df_part1)} rows...")
            self.bq.upload_dataframe(df_part1, BronzeConfig.TABLE_PRICES)
            
        df_part2 = full_df[full_df['time'] >= split_date]
        if not df_part2.empty:
            logger.info(f"📤 Uploading Part 2 (>= 2018): {len(df_part2)} rows...")
            self.bq.upload_dataframe(df_part2, BronzeConfig.TABLE_PRICES)

        logger.info("🎉 Smart Backfill Completed Successfully!")

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize System
    ingestor = BronzeIngestor()
    ingestor.initialize_schema()
    
    # 1. Update Symbols List (Always run this to get fresh VN30/VN100)
    ingestor.ingest_symbol_master()
    
    # 2. Get Target List from BigQuery
    target_symbols = ingestor.bq.query_symbols(filter_condition="is_vn100 = TRUE")
    
    # --- RUN MODE SELECTION ---
    
    # MODE A: Daily Update (Uncomment this for your daily cron job)
    logger.info("Running Daily Mode...")
    ingestor.ingest_prices_daily(target_symbols, lookback_days=3)
    
    # MODE B: Historical Backfill (Uncomment this for one-off history load)
    # WARNING: Takes time. Run only when needed.
    # logger.info("Running Backfill Mode...")
    # ingestor.backfill_history(
    #     symbol_list=target_symbols,
    #     start_str="2010-01-01",
    #     end_str="2012-12-31"  # Adjust year as needed
    # )