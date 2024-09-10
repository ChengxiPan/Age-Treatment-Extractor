import argparse
import logging
from functools import partial
import time
import config
import pandas as pd
from tqdm.auto import tqdm
from utils import *

logging.basicConfig(
    filename='log.log',  # Log file name
    filemode='a',        # Append to the log file, use 'w' to overwrite
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

tqdm.pandas()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ageExtractor', type=str, help='The age extractor function to use')
    parser.add_argument('--treatmentExtractor', type=str, help='The treatment extractor function to use')
    parser.add_argument('--gptModel', type=str, help="Only when treatmentExtractor is gpt_treatment_extractor, can be used")
    return parser.parse_args()

def main():
    #### Init
    args = parse_args()
    df = pd.read_csv(config.METADATA, index_col=0)
    
    #### Age Extraction
    if(args.ageExtractor!=None and args.ageExtractor not in df.columns):
        age_extractor = eval(args.ageExtractor)
        start_time = time.time()
        df[args.ageExtractor] = (df['description'].fillna('') + ' ' + df['transcription'].fillna('')).progress_apply(age_extractor)# fillna(): description and transcription may be empty
        end_time = time.time()
        logging.info(f"AgeExtractor {args.ageExtractor} took {end_time - start_time:.2f} seconds")
        df.to_csv(config.FILS_STORE_PATH)
    else:
        logging.info("Age column already exists")
    
    #### Treatment Extraction
    if(args.treatmentExtractor != None):
        treatment_extractor = eval(args.treatmentExtractor)
        # GPT Extractor
        if(args.treatmentExtractor == 'gpt_treatment_extractor'):
            model = args.gptModel
            if(model not in df.columns):    
                treatment_extractor = partial(treatment_extractor, model=model)
                start_time = time.time()
                df[f'{model}'] = df['transcription'].progress_apply(treatment_extractor)
                end_time = time.time()
                df.to_csv(config.FILS_STORE_PATH)
                logging.info(f"TreatmentExtractor {args.treatmentExtractor}_{args.gptModel} took {end_time - start_time:.2f} seconds")
            else:
                logging.info(f"{model} column already exists")
        else:
            if(args.treatmentExtractor not in df.columns):
                start_time = time.time()
                df[args.treatmentExtractor] = df['transcription'].progress_apply(treatment_extractor)
                end_time = time.time()
                df.to_csv(config.FILS_STORE_PATH)
                logging.info(f"TreatmentExtractor {args.treatmentExtractor} took {end_time - start_time:.2f} seconds")
            else:
                logging.info("Treatment column already exists")
    

if __name__ == '__main__':
    main()    
