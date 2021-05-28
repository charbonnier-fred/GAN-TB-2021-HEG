from keras.datasets import mnist
from numpy import expand_dims, ones
from numpy.random import randint

# Génère n vrais échantillons
def generate_real_samples(n):
  # Charge le jeu d'entrainement depuis MNIST
  (trainX, _), (_, _) = mnist.load_data()
  # transforme les tableaux 2D en 3D en ajoutant un canal supplémentaire
  dataset = expand_dims(trainX, axis=-1)
  # Converti Int et float32
  dataset = dataset.astype("float32")
  # Change l'échelle [0,255] en [0,1]
  dataset = (dataset - 127.5) / 127.5  # Normalize the images to [-1, 1]
  # Choisi aléatoirement des index du dataset
  ix = randint(0, dataset.shape[0], n)
  # Récupère les images sélectionnées
  x = dataset[ix]
  # Ajoute la classe y = 1 pour indiquer qu'il s'agit de vrais échantillons
  y = ones((n, 1))
  return x, y
