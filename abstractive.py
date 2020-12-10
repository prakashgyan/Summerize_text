import re
import os
import pandas as pd
from contraction_mapping import contraction_mapping
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import networkx as nx
from zipfile import ZipFile

train_files = pd.Series(os.listdir('data/dataset/stories_text_summarization_dataset_train/'), name='filenames')

list_stories = pd.Series([open(f'data/dataset/stories_text_summarization_dataset_train/{filename}').read()
                          for filename in train_files], name='story')

# df = pd.DataFrame((train_files,list_stories), columns= ['filename', 'Story' ])
df = pd.merge(train_files, list_stories, left_index=True, right_index=True)
df['passage'] = df.story.apply(lambda x: x.split('@highlight')[0])
df['summary'] = df.story.apply(lambda x: x.split('@highlight')[1:])
print(df.summary[0])
