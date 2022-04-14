import streamlit as st
import praw
import os
from praw.models import MoreComments
import pandas as pd
from transformers import pipeline
import nltk
from langdetect import detect
import numpy as np
from nltk.corpus import stopwords
import string
import requests
from datetime import datetime
from PIL import Image
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
from streamlit import components
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import plotly.express as px
import time

#credentials*********************************************************
reddit_credentials = praw.Reddit(
    user_agent=os.getenv('praw_user_agent'),
    client_id=os.getenv('praw_client_id'),
    client_secret=os.getenv('praw_client_secret'),
    username=os.getenv('praw_username'),
    )

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
    }


# main script
if __name__ == '__main__':
    #Front_end. This paert placed here in order to show something to user while script working
    image = Image.open('Redditpic.png')
    st.image(image, use_column_width=True)

    st.title('Reddit topic Analyzer')

    st.markdown("""
          This app loads topic from Reddit and perform analysis that includes comments classification !
          * **Python libraries:** pandas, streamlit, numpy, matplotlib, seaborn, pyLDAvis, gensim, requests, nltk, praw, transformers, langdetect
          * **Data source:** reddit.com.
          """)
    with st.spinner('Wait for it...'):
        time.sleep(600)

