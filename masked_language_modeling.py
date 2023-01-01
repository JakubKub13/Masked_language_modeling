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

labels = set(data_frame['labels'])
print(labels)

#Pick a label
label = 'business'

texts = data_frame[data_frame['labels'] == label]['text']
texts.head()

np.random.seed(1234)

# Randomly choose a document
i = np.random.choice(texts.shape[0])
doc = texts.iloc[i]

print(textwrap.fill(doc, replace_whitespace=False, fix_sentence_endings=True))

masked_language_model = pipeline('fill-mask')

masked_language_model('Bombardier chief to leave <mask>')
text = 'Shares in <mask> and plane-making giant Bombardier have fallen to 10 year low following the departure of is chief executive'
masked_language_model(text)
text = 'Shares in train and plane-making giant Bombardier have fallen to 10 year low following the departure of is chief <mask>'
pprint(masked_language_model(text))
text = 'Shares in train and plane-making giant Bombardier have fallen to 10 year low following the <mask> of is chief executive'
pprint(masked_language_model(text))

# Exercise: Write a function that automatically masks and replaces words
# In a whole document. Might choose which words to replace based on statistic for example TF-IDF
