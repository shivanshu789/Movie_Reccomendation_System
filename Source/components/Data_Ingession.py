import os
import pandas as pd
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class Data_Ingession_Config:
    Raw_Data_path: str = os.path.join('Artifacts','Raw_data.csv')

class Data_Ingession:
    def __init__(self):
        self.ingession_config = Data_Ingession_Config()

    def Initiate_Data_Ingession(self):
        logging.info('Data Ingession has started')
        try:
            df = pd.read_csv('Notebook/Data/imdb_top_1000.csv')
            print(df.head()) 
            logging.info('Read the file')

            os.makedirs(os.path.dirname(self.ingession_config.Raw_Data_path), exist_ok=True)

            df.to_csv(self.ingession_config.Raw_Data_path, index=False)
            logging.info(f'Data saved to {self.ingession_config.Raw_Data_path}')

        except Exception as e:
            logging.error(f'Failed to read or save file: {e}')

        return self.ingession_config.Raw_Data_path

if __name__ == "__main__":
    data_ingestion = Data_Ingession()
    output_path = data_ingestion.Initiate_Data_Ingession()
    print(f"Raw data path saved to: {output_path}")
