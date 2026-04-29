import numpy as np
import yaml
import pandas as pd
import os
import sys
from dataclasses import dataclass
from src.logger.logging import logging
from src.exception.exception import customexception

@dataclass
class DataIngestionConfig:
    raw_data_path: str = None
    cleaned_data_path: str = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "params.yaml"):
        """Load configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            data_config = config.get('data', {})
            
            return cls(
                raw_data_path=data_config.get('raw_data_path'),
                cleaned_data_path=data_config.get('cleaned_data_path')
            )
        except Exception as e:
            logging.error(f"Error loading params.yaml: {str(e)}")
            raise customexception(e, sys)

class DataIngestion:
    def __init__(self, yaml_config_path: str = "params.yaml"):
        self.ingestion_config = DataIngestionConfig.from_yaml(yaml_config_path)
        self.raw_data_path = self.ingestion_config.raw_data_path
    
    def initiate_data_ingestion(self):
        logging.info("="*60)
        logging.info("DATA INGESTION STARTED")
        logging.info("="*60)
        
        try:
            # Read raw data (Excel or CSV)
            if self.raw_data_path.endswith('.xlsx'):
                df = pd.read_excel(self.raw_data_path)
                logging.info(f"Reading Excel data from: {self.raw_data_path}")
            elif self.raw_data_path.endswith('.csv'):
                df = pd.read_csv(self.raw_data_path)
                logging.info(f"Reading CSV data from: {self.raw_data_path}")
            else:
                raise ValueError(f"Unsupported file format: {self.raw_data_path}")
            
            logging.info(f"✅ Data loaded successfully")
            logging.info(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            logging.info(f"   Columns: {list(df.columns)[:5]}... (showing first 5)")
            
            # Save raw data AS IS (no modifications)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Check file extension and save accordingly
            if self.ingestion_config.raw_data_path.endswith('.csv'):
                df.to_csv(self.ingestion_config.raw_data_path, index=False)
                logging.info(f"💾 Raw data saved as CSV to: {self.ingestion_config.raw_data_path}")
            else:
                df.to_excel(self.ingestion_config.raw_data_path, index=False)
                logging.info(f"💾 Raw data saved as Excel to: {self.ingestion_config.raw_data_path}")
            
            # Also save a copy for next stage (cleaning will modify this)
            os.makedirs(os.path.dirname(self.ingestion_config.cleaned_data_path), exist_ok=True)
            
            if self.ingestion_config.cleaned_data_path.endswith('.csv'):
                df.to_csv(self.ingestion_config.cleaned_data_path, index=False)
                logging.info(f"💾 Initial copy saved as CSV to: {self.ingestion_config.cleaned_data_path}")
            else:
                df.to_excel(self.ingestion_config.cleaned_data_path, index=False)
                logging.info(f"💾 Initial copy saved as Excel to: {self.ingestion_config.cleaned_data_path}")
            
            logging.info("="*60)
            logging.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            logging.info("="*60)
            
            return {
                'raw_data_path': self.ingestion_config.raw_data_path,
                'cleaned_data_path': self.ingestion_config.cleaned_data_path,
                'data_shape': df.shape,
                'columns': list(df.columns)
            }
            
        except Exception as e:
            logging.error(f"❌ Error during data ingestion: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    result = ingestion.initiate_data_ingestion()
    
    print("\n" + "="*60)
    print("DATA INGESTION SUMMARY")
    print("="*60)
    print(f"Raw data saved to: {result['raw_data_path']}")
    print(f"Initial cleaned copy saved to: {result['cleaned_data_path']}")
    print(f"Data shape: {result['data_shape']}")
    print(f"Total columns: {len(result['columns'])}")
    print("="*60)