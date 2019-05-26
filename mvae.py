import pdb
from keras import regularizers
from keras import objectives, backend as K
from keras.layers import Dropout, Reshape, Concatenate, Flatten, Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, RMSprop
import keras
import numpy as np
import os
from sklearn.metrics import precision_score, accuracy_score, precision_recall_fscore_support

class MVAE(object):


    def create(self, max_length, image_embed_size, latent_dim, reg_lambda, fnd_lambda, embed_matrix):

        self.encoder = None
        self.decoder = None
        self.fnd = None
        self.autoencoder = None
        self.embedding_matrix = embed_matrix
        self.vocab_size = self.embedding_matrix.shape[0]
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.reg_lambda = reg_lambda
        self.fnd_lambda = fnd_lambda
        self.image_embed_size = image_embed_size

        input_txt = Input(shape=(self.max_length,), name='input_txt')
        input_img = Input((image_embed_size, ), name='input_img') 

        vae_ce_loss, vae_mse_loss, encoded = self._build_encoder(input_txt, input_img)
        self.encoder = Model(inputs=[input_txt, input_img], outputs=encoded)

        encoded_input = Input(shape=(self.latent_dim,))
        predicted_outcome = self._build_fnd(encoded_input)
        self.fnd = Model(encoded_input, predicted_outcome)

        decoded_txt, decoded_img = self._build_decoder(encoded_input)
        self.decoder = Model(encoded_input, [decoded_txt, decoded_img])

        decoder_output = self._build_decoder(encoded)

        self.autoencoder = Model(inputs=[input_txt, input_img], outputs=[decoder_output[0], decoder_output[1], self._build_fnd(encoded)])
        self.autoencoder.compile(optimizer=Adam(1e-5),
                                 loss=['sparse_categorical_crossentropy', vae_mse_loss, 'binary_crossentropy'],
                                 metrics=['accuracy'])
        self.get_features = K.function([input_txt, input_img], [encoded])
        print self.autoencoder.summary()


    def _build_encoder(self, input_txt, input_img, latent_dim=64):
 
        txt_embed = Embedding(self.vocab_size, 32, input_length=self.max_length, name='txt_embed', trainable=False, weights=[self.embedding_matrix])(input_txt)
        lstm_txt_1 = Bidirectional(LSTM(32, return_sequences=True, name='lstm_txt_1', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda)), merge_mode='concat')(txt_embed)
        lstm_txt_2 = Bidirectional(LSTM(32, return_sequences=False, name='lstm_txt_2', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda)), merge_mode='concat')(lstm_txt_1)
        fc_txt = Dense(32, activation='tanh', name='dense_txt', kernel_regularizer=regularizers.l2(self.reg_lambda))(lstm_txt_2)

        fc_img_1 = Dense(1024, name='fc_img_1', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(input_img)
        fc_img_2 = Dense(32, name='fc_img_2', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(fc_img_1)

        h = Concatenate(axis=-1, name='concat')([fc_txt, fc_img_2])
        h = Dense(64, name='shared', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=0.01)
            return z_mean_ + K.exp(0.5 * z_log_var_) * epsilon

        z_mean = Dense(latent_dim, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_dim, name='z_log_var', activation='linear')(h)

        def vae_mse_loss(x, x_decoded_mean):
            mse_loss = objectives.mse(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return mse_loss + kl_loss
        
        def vae_ce_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_ce_loss, vae_mse_loss, Lambda(sampling, output_shape=(latent_dim,), name='lambda')([z_mean, z_log_var]))


    def _build_decoder(self, encoded):

        dec_fc_txt = Dense(32, name='dec_fc_txt', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(encoded)
        repeated_context = RepeatVector(self.max_length)(dec_fc_txt)
        dec_lstm_txt_1 = LSTM(32, return_sequences=True, activation='tanh', name='dec_lstm_txt_1', kernel_regularizer=regularizers.l2(self.reg_lambda))(repeated_context)
        dec_lstm_txt_2 = LSTM(32, return_sequences=True, activation='tanh', name='dec_lstm_txt_2', kernel_regularizer=regularizers.l2(self.reg_lambda))(dec_lstm_txt_1)
        decoded_txt = TimeDistributed(Dense(self.vocab_size, activation='softmax'), name='decoded_txt')(dec_lstm_txt_2)

        dec_fc_img_1 = Dense(32, name='dec_fc_img_1', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(encoded)
        dec_fc_img_2 = Dense(1024, name='dec_fc_img_2', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(dec_fc_img_1)
        decoded_img = Dense(4096, name='decoded_img', activation='sigmoid')(dec_fc_img_2)

        return decoded_txt, decoded_img


    def _build_fnd(self, encoded):

        h = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(self.fnd_lambda))(encoded)
        h = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(self.fnd_lambda))(h)
        return Dense(1, activation='sigmoid', name='fnd_output')(h)


def train(sequence_length, image_embed_size, latent_dim, reg_lambda, fnd_lambda, path):

    text = np.load('data/train_text.npy')
    im = np.load('data/train_image_embed.npy')
    label = np.load('data/train_label.npy')[:,1]
    
    test_text = np.load('data/test_text.npy')
    test_im = np.load('data/test_image_embed.npy')
    test_label = np.load('data/test_label.npy')[:,1]

    embed_matrix = np.load('data/embedding_matrix.npy')
    vocab_size = embed_matrix.shape[0]
    
    temp = np.zeros((text.shape[0], sequence_length, vocab_size))
    temp[np.expand_dims(np.arange(text.shape[0]), axis=0).reshape(text.shape[0], 1), np.repeat(np.array([np.arange(sequence_length)]), text.shape[0], axis=0), text] = 1
    text_one_hot = temp
    
    temp = np.zeros((test_text.shape[0], sequence_length, vocab_size))
    temp[np.expand_dims(np.arange(test_text.shape[0]), axis=0).reshape(test_text.shape[0], 1), np.repeat(np.array([np.arange(sequence_length)]), test_text.shape[0], axis=0), test_text] = 1
    test_text_one_hot = temp
    
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+'/tb'):
        os.makedirs(path+'/tb')
    if not os.path.exists(path+'/weights'):
        os.makedirs(path+'/weights')
    tensorboard = TensorBoard(log_dir=path+'/tb', write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint(path+'/weights/{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='fnd_output_loss', factor=0.2, patience=6, min_lr=1e-7)

    model = MVAE()
    model.create(sequence_length, image_embed_size, latent_dim, reg_lambda, fnd_lambda, embed_matrix)
    model.autoencoder.fit(x=[text, im], y={'decoded_txt': np.expand_dims(text, -1), 'decoded_img':im, 'fnd_output': label},
                          batch_size=128, epochs=300, callbacks=[checkpoint, tensorboard, reduce_lr], shuffle=True,
                          validation_data=([test_text, test_im], {'decoded_txt': np.expand_dims(test_text, -1), 'decoded_img':test_im, 'fnd_output': test_label}))


def save_features(sequence_length, image_embed_size, latent_dim, reg_lambda, fnd_lambda, path):
    
    test_text = np.load('../data/test_text.npy')
    test_im = np.load('../data/test_image_embed.npy')

    embed_matrix = np.load('../data/embedding_matrix.npy')
    vocab_size = embed_matrix.shape[0]
    
    model = MVAE()
    model.create(sequence_length, image_embed_size, latent_dim, reg_lambda, fnd_lambda, embed_matrix)
    model.autoencoder.load_weights(path+'/weights/286.hdf5')
    
    if not os.path.exists(path+'/features'):
        os.makedirs(path+'/features')

    learnt_features = np.array([]).reshape(0, 64) 
    for i in range(test_text.shape[0]):
        text_batch = test_text[i:i+1]
        im_batch = test_im[i:i+1]
        batch = model.get_features([text_batch, im_batch])[0]
        learnt_features = np.concatenate([learnt_features, batch])
    np.save(path+'/features/vae_fnd', learnt_features)


def test(sequence_length, image_embed_size, latent_dim, reg_lambda, fnd_lambda, path):

    test_text = np.load('data/test_text.npy')
    test_im = np.load('data/test_image_embed.npy')
    test_label = np.load('data/test_label.npy')[:,1]

    embed_matrix = np.load('data/embedding_matrix.npy')
    vocab_size = embed_matrix.shape[0]

    model = MVAE()
    model.create(sequence_length, image_embed_size, latent_dim, reg_lambda, fnd_lambda, embed_matrix)
    model.autoencoder.load_weights(path+'/weights/224.hdf5')
    for i in range(10):
        pred = model.autoencoder.predict([test_text, test_im])[-1]
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        print accuracy_score(test_label, pred)
        print precision_recall_fscore_support(test_label, pred)

    pdb.set_trace()


if __name__ == '__main__':

    train(20, 4096, 64, 0.05, 0.3, 'models/vae_fnd_0.05_0.3')
    test(20, 4096, 64, 0.05, 0.3, 'models/vae_fnd_0.05_0.3')
    save_features(20, 4096, 64, 0.05, 0.3, '../models/vae_fnd_0.05_0.3')
