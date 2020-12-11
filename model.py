import tensorflow as tf
import tensorflow.keras.layers as L
from attention import AttentionLayer


def build_model(pass_max_len, x_vocab_size, y_vocab_size, encoder_layers=3, decoder_layers=1,
                encoder_dim=500, decoder_dim=500, layer_type='LSTM'):
    if layer_type == 'LSTM':
        # Encoder
        enc_inputs = L.Input(shape=(pass_max_len,))
        enc_embeddings = L.Embedding(x_vocab_size, encoder_dim)(enc_inputs)

        enc_rnn_layers = [L.LSTM(encoder_dim, return_sequences=True, return_state=True) for i in range(encoder_layers)]
        enc_rnn_output = enc_embeddings
        for layer in enc_rnn_layers:
            enc_rnn_output, enc_state_h, enc_state_c = layer(enc_rnn_output)

        # Decoder
        dec_inputs = L.Input(shape=(None,))
        dec_embeddings = L.Embedding(y_vocab_size, decoder_dim)(dec_inputs)

        dec_rnn_layers = [L.LSTM(decoder_dim, return_sequences=True, return_state=True) for i in range(decoder_layers)]
        dec_rnn_output = dec_embeddings
        for layer in dec_rnn_layers:
            dec_rnn_output, dec_state_h, dec_state_c = layer(dec_rnn_output, initial_state=[enc_state_h, enc_state_c])

        # Attention layer
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([enc_rnn_output, dec_rnn_output])

        # concat
        dec_concat_op = L.Concatenate(axis=-1, name='concat_layer')([dec_rnn_output, attn_out])

        # Dense
        dec_dense_op = L.TimeDistributed(L.Dense(y_vocab_size, activation='softmax'))(dec_concat_op)

        model = tf.keras.Model([enc_inputs, dec_inputs], dec_dense_op)

        print(model.summary())

        return model




