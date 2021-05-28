from datetime import datetime
from model import GAN
import os
import tensorflow as tf
import utils

# Nombre d'époques
epochs = 3000
# Taille des lots
batch_size = 1
# Taux d'apprentissage des générateurs
g_learning_rate = 0.0002
# Taux d'apprentissage des discriminateurs
d_learning_rate = 0.000005
# Decay
g_learning_rate_decay = g_learning_rate / 200000
d_learning_rate_decay = d_learning_rate / 200000

# Répertoire de log
logdir="./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Répertoire Voix A pour l'entrainement
dir_wavs_A_train = "wavs/voix_A_train/"
# Répertoire Voix B pour l'entrainement
dir_wavs_B_train = "wavs/voix_B_train/"
# Répertoire Voix A pour le test
dir_wavs_A_test = "wavs/voix_A_test/"
# Répertoire Voix B pour le test
dir_wavs_B_test = "wavs/voix_B_test/"
# Répertoire de cache
dir_cache = "cache/"
dir_model = "model/"
path_logf0s_norm = os.path.join(dir_cache, 'logf0s_normalization.npz')
path_mcep_norm = os.path.join(dir_cache, 'mcep_normalization.npz')
path_coded_sps_A_norm = os.path.join(dir_cache, "coded_sps_A_norm.pickle")
path_coded_sps_B_norm = os.path.join(dir_cache, "coded_sps_B_norm.pickle")
# Taux d'échantillonnage (Hz)
sr = 22000
# Durée d'une trame (ms)
frame_period = 5.0
# Nombre de dimensions MCEPs souhaitées
num_mcep = 36
# Nombre de trames utilisées pour l'entrainement
# par échantillon
n_frames = 128

print("debut preprocessing")
# Preprocessing pour l'extraction des caractéristiques audios
coded_sps_A_norm, coded_sps_B_norm = utils.preprocessing(dir_wavs_A_train, dir_wavs_B_train, dir_cache, sr, frame_period, num_mcep, 
                                                         path_logf0s_norm, path_mcep_norm, path_coded_sps_A_norm, path_coded_sps_B_norm)
print("fin preprocessing")
# Instanciation du GAN
gan = GAN(epochs, batch_size, d_learning_rate, g_learning_rate, g_learning_rate_decay, d_learning_rate_decay,
          logdir, num_mcep, n_frames, sr, frame_period, dir_wavs_A_test, dir_wavs_B_test, path_logf0s_norm, path_mcep_norm, dir_model)

# Entrainement du GAN
gan.train(coded_sps_A_norm, coded_sps_B_norm)
