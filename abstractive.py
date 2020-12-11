import os
import pandas as pd
from text_cleaner import story_cleaner, summary_cleaner
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_model
import plotly.express as px

train_files = pd.Series(os.listdir('data/dataset/stories_text_summarization_dataset_train/'), name='filenames')

list_stories = pd.Series([open(f'data/dataset/stories_text_summarization_dataset_train/{filename}').read()
                          for filename in train_files], name='story')
# df = pd.DataFrame((train_files,list_stories), columns= ['filename', 'Story' ])
df = pd.merge(train_files, list_stories, left_index=True, right_index=True)
df = df[:1000]

# print(time.time())
# df['story'] = df.story.apply(text_cleaner)
print(time.time())
df['passage'] = df.story.apply(lambda x: x.split('@highlight')[0])
df['summary'] = df.story.apply(lambda x: x.split('@highlight')[1:])
df['passage'] = df.passage.apply(story_cleaner)
df['summary'] = df.summary.apply(summary_cleaner)
print(time.time())

summ_max_len = max(df.summary.apply(lambda x: len(x.split(' '))))
pass_max_len = max(df.passage.apply(lambda x: len(x)))

x_train, x_val, y_train, y_val = train_test_split(df.passage, df.summary, test_size=0.15, shuffle=True)

# print(x_train[:10].values)

x_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
y_tokenizer = tf.keras.preprocessing.text.Tokenizer()

x_tokenizer.fit_on_texts(x_train.values)
y_tokenizer.fit_on_texts(y_train.values)

x_train_seq = x_tokenizer.texts_to_sequences(x_train)
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

y_train_seq = y_tokenizer.texts_to_sequences(y_train)
y_val_seq = y_tokenizer.texts_to_sequences(y_val)

x_train_seq_pad = pad_sequences(x_train_seq, padding='post', maxlen=pass_max_len)
x_val_seq_pad = pad_sequences(x_val_seq, padding='post', maxlen=pass_max_len)
y_train_seq_pad = pad_sequences(y_train_seq, padding='post', maxlen=summ_max_len)
y_val_seq_pad = pad_sequences(y_val_seq, padding='post', maxlen=summ_max_len)

x_vocab_size = len(x_tokenizer.word_index) + 1
y_vocab_size = len(y_tokenizer.word_index) + 1

model = build_model(pass_max_len, x_vocab_size, y_vocab_size)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

history = model.fit([x_train_seq_pad, y_train_seq_pad[:, :-1]],
                    y_train_seq_pad.reshape(y_train_seq_pad.shape[0], y_train_seq_pad.shape[1], 1)[:, 1:],
                    epochs=25, batch_size=4,
                    validation_data=([x_val_seq_pad, y_val_seq_pad[:, :-1]],
                                     y_val_seq_pad.reshape(y_val_seq_pad.shape[0], y_val_seq_pad.shape[1], 1)[:, 1:])
                    )

fig = px.line(history.history)
fig.show()