#https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
#!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv
#!pip install transformers

import numpy as np
import pandas as pd
import textwrap
from pprint import pprint
from transformers import pipeline

data_frame = pd.read_csv('bbc_text_cls.csv')
data_frame.head()
