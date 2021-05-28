import io
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, Reshape
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
import time

def discriminator():
  # Entrée à 28 x 28 pixels et 1 canal
  inp = Input(shape=(28,28,1), name='input_sample')
  x = inp
  # Couche convolutive
  x = Conv2D(64, (5,5), strides=(2, 2), padding="same")(x)
  x = LeakyReLU()(x)
  x = Dropout(0.3)(x)
  # Couche convolutive
  x = Conv2D(128, (5,5), strides=(2, 2), padding="same")(x)
  x = LeakyReLU()(x)
  x = Dropout(0.3)(x)
  x = Flatten()(x)
  # Couche de sortie
  last = Dense(1,name="output")(x)
  return Model(inputs=inp, outputs=last)

def generator(latent_dim):
  # Entrée à "latent_dim" valeurs
  inp = Input(shape=(latent_dim,), name='input_sample')
  x = inp
  x = Dense(7 * 7 * 256, use_bias=False)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  x = Reshape((7, 7, 256))(x)
  x = Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  x = Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  # Ajout de la sortie avec la fonction d'activation sigmoid
  last = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh', name="output")(x)
  return Model(inputs=inp, outputs=last)

class GAN(object):
  # Constructeur
  def __init__(self, epochs, batch_size, latent_dim, d_learning_rate, g_learning_rate, logdir):
    self.epochs = epochs
    self.batch_size = batch_size
    self.latent_dim = latent_dim
    # Instanciation de l'optimiseur du discriminateur
    self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=d_learning_rate)
    # Instanciation de l'optimiseur du générateur
    self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=g_learning_rate)
    self.logdir = logdir
    self.discriminator = discriminator()
    self.generator = generator(latent_dim)
    # Instanciation de la fonction de perte du discrinateur
    self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Mesures de performance
    self.g_accuracy = tf.keras.metrics.BinaryAccuracy()
    self.d_accuracy = tf.keras.metrics.BinaryAccuracy()
    self.d_real_accuracy = tf.keras.metrics.BinaryAccuracy()
    self.d_generated_accuracy = tf.keras.metrics.BinaryAccuracy()
  
  def generator_loss(self, generated_output):
    # Calcul de la perte du générateur à l'aide des faux échantillons
    # ayant été classifiés comme "vrai" par le discriminateur
    return self.loss_fn(tf.ones_like(generated_output), generated_output)

  def discriminator_loss(self, real_output, generated_output):
    # Calcul de la perte du discriminateur à l'aide des vrais échantillons
    # ayant été classifiés comme "vrai" et des faux échantillons ayant 
    # été classifiés comme "faux"
    real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
    generated_loss = self.loss_fn(tf.zeros_like(generated_output), generated_output)
    total_loss = real_loss + generated_loss
    return total_loss
  
  def generator_accuracy(self, generated_output):
    # Calcul de la performance du générateur à l'aide des faux échantillons
    # ayant été classifiés comme "vrai" par le discriminateur
    self.g_accuracy.reset_states()
    return self.g_accuracy(tf.ones_like(generated_output), generated_output)
  
  def discriminator_accuracy(self, real_output, generated_output):
    # Calcul de la performance du discriminateur à l'aide des échantillons réels
    # ayant été classifiés comme "vrai" et des faux échantillons générés par
    # le générateur ayant été classifiés comme "faux"
    self.d_accuracy.reset_states()
    return self.d_accuracy(tf.concat([tf.ones_like(real_output), tf.zeros_like(generated_output)], 0), 
      tf.concat([real_output, generated_output], 0))
  
  def discriminator_real_accuracy(self, real_output):
    # Calcul de la performance du discriminateur à classifier 
    # les vrais échantillons à "vrai"
    self.d_real_accuracy.reset_states()
    return self.d_real_accuracy(tf.ones_like(real_output), real_output)
  
  def discriminator_generated_accuracy(self, generated_output):
    # Calcul de la performance du discriminateur à classifier 
    # les faux échantillons à "faux"
    self.d_generated_accuracy.reset_states()
    return self.d_generated_accuracy(tf.zeros_like(generated_output), generated_output)

  def train_step(self, real_samples):
    # Génére les points de l'espace latent
    random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # Génère les faux échantillons
      generated_samples = self.generator(random_latent_vectors, training=True)

      # Classifie les vrais échantillons
      real_output = self.discriminator(real_samples, training=True)
      # Classifie les faux échantillons
      generated_output = self.discriminator(generated_samples, training=True)

      # Calcul la perte du générateur
      gen_loss = self.generator_loss(generated_output)
      # Calcul la perte du discriminateur
      disc_loss = self.discriminator_loss(real_output, generated_output)

      # Calcul des performances du générateur
      gen_acc = self.generator_accuracy(generated_output)
      # Calcul des performances du discriminateur
      disc_acc = self.discriminator_accuracy(real_output, generated_output)
      # Calcul des performances du discriminateur avec les vrais échantillons
      disc_real_acc = self.discriminator_real_accuracy(real_output)
      # Calcul des performances du discriminateur avec les faux échantillons
      disc_gen_acc = self.discriminator_generated_accuracy(generated_output)

    # Récupère les gradients des variables entraînables par rapport à la perte
    gradients_of_generator = gen_tape.gradient(gen_loss, 
      self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, 
      self.discriminator.trainable_variables)

    # Effectue une étape de descente de gradient en mettant à jour la valeur 
    # des variables pour minimiser la perte
    self.g_optimizer.apply_gradients(zip(gradients_of_generator, 
      self.generator.trainable_variables))
    self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, 
      self.discriminator.trainable_variables))

    return gen_loss, disc_loss, gen_acc, disc_acc, disc_real_acc, disc_gen_acc

  def train(self, dataset):
    file_writer = tf.summary.create_file_writer(self.logdir)
    file_writer.set_as_default()

    time_list = []

    # Boucle sur les époques
    for epoch in range(self.epochs):
      start_time = time.time()
      # Boucle sur les lots du jeu de données
      for real_samples in dataset:
        # Entraine le GAN à l'aide d'un lot de vrais échantillons
        gen_loss, disc_loss, gen_acc, disc_acc, disc_real_acc, disc_gen_acc = self.train_step(real_samples)

      wall_time_sec = time.time() - start_time
      time_list.append(wall_time_sec)

      template = 'Epoch {}, Generator loss {}, Discriminator Loss {}, Generator accuracy {}, Discriminator accuracy {}, Discriminator real accuracy {}, Discriminator fake accuracy {}'
      print (template.format(epoch, gen_loss, disc_loss, gen_acc, disc_acc, disc_real_acc, disc_gen_acc))

      # Logue les différentes pertes
      tf.summary.scalar('1. Generator loss', gen_loss, step=epoch)
      tf.summary.scalar('2. Discriminator loss', disc_loss, step=epoch)
      # Logue les différentes performances
      tf.summary.scalar('3. Generator accuracy', gen_acc, step=epoch)
      tf.summary.scalar('4. Discriminator accuracy', disc_acc, step=epoch)
      tf.summary.scalar('5. Discriminator real accuracy', disc_real_acc, step=epoch)
      tf.summary.scalar('6. Discriminator fake accuracy', disc_gen_acc, step=epoch)
      # Récupère 2 lots de vrais échantillons
      #real_samples = np.concatenate([x for x in dataset.take(2)], axis=0)
      random_latent_vectors = tf.random.normal(shape=(25, self.latent_dim))
      # Récupère 25 faux échantillons
      generated_samples = np.array(self.generator(random_latent_vectors))
      # Dessine le résultat
      fig = pyplot.figure()
      for i in range(25):
        # define subplot
        pyplot.subplot(5, 5, 1 + i)
        # turn off axis labels
        pyplot.axis('off')
        # plot single image
        pyplot.imshow(generated_samples[i, :, :, 0], cmap='gray_r')
      tf.summary.image("Output generator", plot_to_image(fig), step=(epoch+1))

    return time_list
  
def plot_to_image(figure): 
  # Sauvegarde le diagramme en mémoire au format PNG 
  buf = io.BytesIO() 
  pyplot.savefig(buf, format='png') 
  pyplot.close(figure) 
  buf.seek(0) 
  # Convertir le tampon PNG en image TF
  image = tf.image.decode_png(buf.getvalue(), channels=4) 
  # Ajoute la dimension lots 
  image = tf.expand_dims(image, 0) 
  return image 