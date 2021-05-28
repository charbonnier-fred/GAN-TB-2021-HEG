from datetime import datetime
from model import GAN
from sample import generate_real_samples
import tensorflow as tf

# Nombre d'époques
epochs = 1000
# Taille des lots
batch_size = 256
# Taille de l'espace latent
latent_dim = 100
# Taux d'apprentissage du discriminateur
d_learning_rate = 0.0001
# Taux d'apprentissage du générateur
g_learning_rate = 0.0001
# Emplacement du répertoire de logs
logdir="./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Instanciation du GAN
gan = GAN(epochs, batch_size, latent_dim, d_learning_rate, g_learning_rate, logdir)

# Préparation du jeu de données
real_samples, _ = generate_real_samples(3000)
dataset = tf.data.Dataset.from_tensor_slices(real_samples.astype("float32"))
# Mélange et mise en lots des données
dataset = dataset.shuffle(buffer_size=3000).batch(batch_size)

# Entrainement du GAN
gan.train(dataset)
