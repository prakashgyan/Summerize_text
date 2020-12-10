import re
import os
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import networkx as nx
from zipfile import ZipFile


nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')
url = 'http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'
# print(not os.path.exists('data/glove.6B.zip'))
if not os.path.exists('data/glove.6B.zip'):
    os.system("curl 'http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip' -o data/glove.6B.zip")

if not os.path.exists('data/glove.6B/glove.6B.100d.txt'):
    with ZipFile('data/glove.6B.zip', 'r') as data:
        print('Extracting.. GloVe Files')
        data.extractall('data/glove.6B')
        print('Extraction Done')

random.seed(42)
train_files = os.listdir('data/dataset/stories_text_summarization_dataset_train/')
random_100_filenames = random.choices(train_files, k=10)

list_stories = [open(f'data/dataset/stories_text_summarization_dataset_train/{filename}').read().split('@highlight')[0]
                for filename in random_100_filenames]

# with open('data/dataset/stories_text_summarization_dataset_train/941e109148d5dab20ca0eedbc030a9c80caee816.story',
#           'r') as story:
#     s = story.read()
#
# s = s.split('@highlight')[0]

list_stories_sentences = [sent_tokenize(s) for s in list_stories]
# list_stories_sentences.append(sent_tokenize(s))
# list_stories_sentences = [y for x in list_stories_sentences for y in x]  # flatten list

# print(list_stories_sentences)

# !wget 'http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'


word_embeddings = {}
f = open('data/glove.6B/glove.6B.100d.txt', encoding='utf-8')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

len(word_embeddings)
#
# # Cleaning
story_clean_sentences = [[re.sub(r'\([^)]*\)', '', sentence) for sentence in sentences]
                   for sentences in list_stories_sentences]
story_clean_sentences = [[re.sub(r'[^a-zA-Z]', ' ', sentence) for sentence in sentences]
                   for sentences in story_clean_sentences]
story_clean_sentences = [[s.lower() for s in sentences] for sentences in story_clean_sentences]

# print(story_clean_sentences)


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# remove stopwords from the sentences
story_clean_sentences = [[remove_stopwords(r.split()) for r in clean_sentences]
                         for clean_sentences in story_clean_sentences]
# story_vectors = []
list_sim_mat = []
for story in story_clean_sentences:
    sentence_vectors = []
    for i in story:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    sim_mat = np.zeros([len(story), len(story)])

    for i in range(len(story)):
        for j in range(len(story)):
            if i != j:
                sim_mat[i][j] = \
                cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[
                    0, 0]
    list_sim_mat.append(sim_mat)
    # story_vectors.append(sentence_vectors)
# similarity matrix
# list_sim_mat = []
# for sentences in story_clean_sentences:
#
#     sim_mat = np.zeros([len(story), len(story)])
#
# for i in range(len(story)): for j in range(len(story)): if i != j: sim_mat[i][j] = cosine_similarity(
# sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[ 0, 0] list_sim_mat.append(sim_mat)

# print(list_sim_mat)

list_nx_graph = [nx.from_numpy_array(sim_mat) for sim_mat in list_sim_mat]
list_scores = [nx.pagerank(nx_graph) for nx_graph in list_nx_graph]

# list_ranked_sentences = [sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True) for sentences in
# story_clean_sentences] for i in range(3): print(ranked_sentences[i][1])
#
list_scores_items = [scores.items() for scores in list_scores]
#
list_summary_sent_codes = [sorted(sorted(scores_items, key=lambda x: x[-1], reverse=True)[:3], key=lambda x: x[0])
                     for scores_items in list_scores_items]
# summary_sent = sorted(summary_sent, key=lambda x: x[0])
#
list_summary_sentences = [(z, [y[a] for a, b in x]) for x, y, z in
                          zip(list_summary_sent_codes, list_stories_sentences, random_100_filenames)]
# result = pd.DataFrame([random_100_filenames,])
# for x, y in summary_sent:
#     summary.append(sentences[x])
#
# print(summary)

for filename, summary in list_summary_sentences:
    print(f'{filename.split(".")[0]} : {summary}')



