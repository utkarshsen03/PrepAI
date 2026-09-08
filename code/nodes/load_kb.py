import os
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Set up basic logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_knowledge_data(kb_path):
    try:
        logger.info("Starting fetch_knowledge_data process.")
        # response = requests.get(os.getenv("CONFIG_PATH"))
        # configs = response.json()
        # print(configs)
        # kb_path = configs.get(str(id))
        # print(kb_path)
        kb_df = pd.read_excel(kb_path,engine='openpyxl')
        # kb_df = kb_df.drop(columns=["Answer"])
        # kb_df = kb_df[kb_df["Position"] == position]
        logging.info(f"Knowledge base data fetched successfully with {len(kb_df)} entries.")
        return len(kb_df), kb_df.to_csv(index=False)
    except Exception as e:
        logger.exception(f"Error occurred while fetching knowledge base data: {e}")
        raise  e