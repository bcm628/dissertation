#TODO: finish this

batch_size = 20
epochs = 100
#TODO: check these
gamma = 10
theta = 1

modality = 'acoustic'
mod_dim = 74
proj_dim = 40
seq_len = 50
save_name = None

#LSTM param
lstm_dim = 128
lstm_layers = 3
bidirectional = True

#RNN params
rnn_dim = 64
rnn_layers = 2


emo_labels = ['happy', 'sad', 'angry']

#iemocap_path = 'C:/Users/bcmye/PycharmProjects/dissertation/Data/IEMOCAP_aligned'
iemocap_path = '/content/drive/My Drive/Colab Notebooks'
mosei_path = 'C:/Users/bcmye/PycharmProjects/CMU-MultimodalSDK/data/MOSEI_aligned/'
