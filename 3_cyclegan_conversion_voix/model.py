import io
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization, Conv1D, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, Reshape
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import librosa.display
from matplotlib import pyplot
import numpy as np
import os
import random
import tensorflow as tf
import time
import utils

# Couche convolutive 2D
def conv2d_layer(inputs, filters, kernel_size, strides, padding='same', 
                 activation=None, kernel_initializer=None, name=None):
  conv_layer = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding, activation=activation, 
                      kernel_initializer=kernel_initializer, name=name)(inputs)
  
  return conv_layer

# Activation GLU
def gated_linear_layer(inputs, gates, name=None):
  activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)
  
  return activation

# Bloc Downsample (2D)
def downsample2d_block(inputs, filters, kernel_size, strides, name_prefix='downsample2d_block_'):
  h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                    name=name_prefix + 'h1_conv')
  h1_norm = InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')(h1)
  h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                          name=name_prefix + 'h1_gates')
  h1_norm_gates = InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')(h1_gates)
  h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

  return h1_glu

def discriminator(num_mcep, n_frames):
  # Entrée
  inp = Input(shape=(num_mcep, n_frames, 1), name='input_sample')
  inputs = inp

  h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], 
                    strides = [1, 1], activation = None, name = 'h1_conv')
  h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], 
                          strides = [1, 1], activation = None, name = 'h1_conv_gates')
  h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

  # Downsample
  d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = [3, 3], 
                          strides = [2, 2], name_prefix = 'downsample2d_block1_')
  d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = [3, 3], 
                          strides = [2, 2], name_prefix = 'downsample2d_block2_')
  d3 = downsample2d_block(inputs = d2, filters = 1024, kernel_size = [3, 3], 
                          strides = [2, 2], name_prefix = 'downsample2d_block3_')
  d4 = downsample2d_block(inputs=d3, filters=1024, kernel_size=[1, 5], 
                          strides=[1, 1], name_prefix='downsample2d_block4_')

  # Output
  o1 = conv2d_layer(inputs=d4, filters=1, kernel_size=[1, 3], 
                    strides=[1, 1], activation=tf.nn.sigmoid, name='out_1d_conv')

  return Model(inputs=inp, outputs=o1)

# Couche convolutive 1D
def conv1d_layer(inputs, filters, kernel_size, strides=1, padding='same',
                 activation=None, kernel_initializer=None, name=None):
  conv_layer = Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding, activation=activation,
                      kernel_initializer=kernel_initializer, name=name)(inputs)

  return conv_layer

# ResBlock
def residual1d_block(inputs, filters, kernel_size, strides, name_prefix='residule_block_'):
  h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, 
                    strides=strides, activation=None, name=name_prefix + 'h1_conv')
  h1_norm = InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')(h1)
  h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, 
                          strides=strides, activation=None, name=name_prefix + 'h1_gates')
  h1_norm_gates = InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')(h1_gates)
  h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
  h2 = conv1d_layer(inputs=h1_glu, filters=filters // 2, kernel_size=kernel_size, 
                    strides=strides, activation=None, name=name_prefix + 'h2_conv')
  h2_norm = InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h2_norm')(h2)

  h3 = inputs + h2_norm

  return h3

# Bloc Upsample (2D)
def upsample2d_block(inputs, filters, kernel_size, strides, shuffle_size=2, name_prefix='upsample1d_block_'):
  h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
  h1_shuffle = tf.nn.depth_to_space(h1, block_size=2, name='h1_shuffle')
  h1_norm = InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')(h1_shuffle)

  h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                          name=name_prefix + 'h1_gates')
  h1_shuffle_gates = tf.nn.depth_to_space(h1_gates, block_size=2, name='h1_shuffle')
  h1_norm_gates = InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')(h1_shuffle_gates)

  h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

  return h1_glu

def generator(num_mcep):
  # Entrée
  inp = Input(shape=(num_mcep, None, 1), name='input_sample')
  inputs = inp

  res_filter = 512
  batch_size = 1

  h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5, 15], 
                    strides = [1, 1], activation = None, name = 'h1_conv')
  h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5, 15], 
                          strides = 1, activation = None, name = 'h1_conv_gates')
  h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

  # Downsample
  d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = 5, 
                          strides = 2, name_prefix = 'downsample1d_block1_')
  d2 = downsample2d_block(inputs = d1, filters = 256, kernel_size = 5, 
                          strides = 2, name_prefix = 'downsample1d_block2_')

  # Reshape
  d3 = tf.squeeze(tf.reshape(d2, shape=(batch_size, 1, -1, 2304)), axis=1)

  # 1x1 Conv
  d3 = conv1d_layer(inputs=d3, filters=256, kernel_size = 1, 
                    strides = 1, activation = None, name = '1x1_down_conv1d')
  d3 = InstanceNormalization(epsilon=1e-6, name='d3_norm')(d3)

  # Residual blocks
  r1 = residual1d_block(inputs = d3, filters = res_filter, kernel_size = 3, 
                        strides = 1, name_prefix = 'residual1d_block1_')
  r2 = residual1d_block(inputs = r1, filters = res_filter, kernel_size = 3, 
                        strides = 1, name_prefix = 'residual1d_block2_')
  r3 = residual1d_block(inputs = r2, filters = res_filter, kernel_size = 3, 
                        strides = 1, name_prefix = 'residual1d_block3_')
  r4 = residual1d_block(inputs = r3, filters = res_filter, kernel_size = 3, 
                        strides = 1, name_prefix = 'residual1d_block4_')
  r5 = residual1d_block(inputs = r4, filters = res_filter, kernel_size = 3, 
                        strides = 1, name_prefix = 'residual1d_block5_')
  r6 = residual1d_block(inputs = r5, filters = res_filter, kernel_size = 3, 
                        strides = 1, name_prefix = 'residual1d_block6_')

  # 1x1 Conv
  r6 = conv1d_layer(r6, filters = 2304, kernel_size = 1, 
                    strides = 1, activation = None, name = '1x1_up_conv1d')
  r6 = InstanceNormalization(epsilon=1e-6, name='r6_norm')(r6)

  # Reshape
  r6 = tf.reshape(tf.expand_dims(r6, axis=1), shape=(batch_size, 9, -1, 256))

  # Upsample
  u1 = upsample2d_block(inputs = r6, filters = 1024, kernel_size = 5, 
                        strides = 1, name_prefix = 'upsample1d_block1_')
  u2 = upsample2d_block(inputs = u1, filters = 512, kernel_size = 5, 
                        strides = 1, name_prefix = 'upsample1d_block2_')

  # Output
  o1 = conv2d_layer(inputs = u2, filters = 1, kernel_size = [5, 15], 
                    strides = [1, 1], activation = None, name = 'o1_conv')

  return Model(inputs=inp, outputs=o1)

class GAN(object):
  # Constructeur
  def __init__(self, epochs, batch_size, d_learning_rate, g_learning_rate, g_learning_rate_decay, d_learning_rate_decay, logdir,
               num_mcep, n_frames, sr, frame_period, dir_wavs_A_test, dir_wavs_B_test, path_logf0s_norm, path_mcep_norm, dir_model):
    self.epochs = epochs
    self.batch_size = batch_size
    self.d_learning_rate = d_learning_rate
    self.g_learning_rate = g_learning_rate
    # Instanciation de l'optimiseur du discriminateur
    self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=d_learning_rate, beta_1=0.5)
    # Instanciation de l'optimiseur du générateur
    self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=g_learning_rate, beta_1=0.5)
    self.g_learning_rate_decay = g_learning_rate_decay
    self.d_learning_rate_decay = d_learning_rate_decay
    self.logdir = logdir
    self.discriminator_B = discriminator(num_mcep, n_frames)
    self.discriminator_A = discriminator(num_mcep, n_frames)
    self.generator_AB = generator(num_mcep)
    self.generator_BA = generator(num_mcep)
    # Instanciation de la fonction de perte du discrinateur
    self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Mesures de performance
    self.g_accuracy = tf.keras.metrics.BinaryAccuracy()
    self.d_accuracy = tf.keras.metrics.BinaryAccuracy()
    self.d_real_accuracy = tf.keras.metrics.BinaryAccuracy()
    self.d_generated_accuracy = tf.keras.metrics.BinaryAccuracy()
    self.num_mcep = num_mcep
    self.n_frames = n_frames
    self.sr = sr
    self.frame_period = frame_period
    self.dir_wavs_A_test = dir_wavs_A_test
    self.dir_wavs_B_test = dir_wavs_B_test
    logf0s_normalization = np.load(path_logf0s_norm)
    self.log_f0s_mean_A = logf0s_normalization['mean_A']
    self.log_f0s_std_A = logf0s_normalization['std_A']
    self.log_f0s_mean_B = logf0s_normalization['mean_B']
    self.log_f0s_std_B = logf0s_normalization['std_B']
    mcep_normalization = np.load(path_mcep_norm)
    self.coded_sps_A_mean = mcep_normalization['mean_A']
    self.coded_sps_A_std = mcep_normalization['std_A']
    self.coded_sps_B_mean = mcep_normalization['mean_B']
    self.coded_sps_B_std = mcep_normalization['std_B']
    self.lambda_cycle = 10.0
    self.lambda_identity = 5.0
    self.dir_model = dir_model
  
  def generator_loss(self, generated_output):
    # Calcul de la perte du générateur à l'aide des faux échantillons
    # ayant été classifiés comme "vrai" par le discriminateur
    return tf.reduce_mean(tf.square(tf.ones_like(generated_output) - generated_output))

  def discriminator_loss(self, real_output, generated_output):
    # Calcul de la perte du discriminateur à l'aide des vrais échantillons
    # ayant été classifiés comme "vrai" et des faux échantillons ayant 
    # été classifiés comme "faux"
    real_loss = tf.reduce_mean(tf.square(tf.ones_like(real_output) - real_output))
    generated_loss = tf.reduce_mean(tf.square(tf.zeros_like(generated_output) - generated_output))
    total_loss = (real_loss + generated_loss) / 2.0
    return total_loss
  
  def calc_cycle_loss(self, real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return loss1
  
  def identity_loss(self, real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return loss
  
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

  def train_step(self, dataset_A_batch, dataset_B_batch):

    with tf.GradientTape() as gen_tape:
      # Génère les faux échantillons de la voix B depuis
      # les vrais échantillons de la voix A
      fake_samples_B = self.generator_AB(dataset_A_batch, training=True)
      cycled_samples_A = self.generator_BA(tf.squeeze(fake_samples_B, axis=-1), training=True)
      
      fake_samples_A = self.generator_BA(dataset_B_batch, training=True)
      cycled_samples_B = self.generator_AB(tf.squeeze(fake_samples_A, axis=-1), training=True)

      same_samples_A = tf.dtypes.cast(tf.squeeze(self.generator_BA(dataset_A_batch, training=True), axis=-1), tf.float64)
      same_samples_B = tf.dtypes.cast(tf.squeeze(self.generator_AB(dataset_B_batch, training=True), axis=-1), tf.float64)

      # Classifie les faux échantillons de la voix B
      d_fake_output_A = self.discriminator_A(fake_samples_A, training=True)
      # Classifie les faux échantillons de la voix B
      d_fake_output_B = self.discriminator_B(fake_samples_B, training=True)
      
      # for the second step adverserial loss
      d_fake_output_cycle_A = self.discriminator_A(cycled_samples_A)
      d_fake_output_cycle_B = self.discriminator_B(cycled_samples_B)
      
      cycle_A_loss = self.calc_cycle_loss(dataset_A_batch, tf.dtypes.cast(tf.squeeze(cycled_samples_A, axis=-1), tf.float64))
      cycle_B_loss = self.calc_cycle_loss(dataset_B_batch, tf.dtypes.cast(tf.squeeze(cycled_samples_B, axis=-1), tf.float64))
      total_cycle_loss = cycle_A_loss + cycle_B_loss

      identity_A_loss = self.identity_loss(dataset_A_batch, same_samples_A)
      identity_B_loss = self.identity_loss(dataset_B_batch, same_samples_B)
      total_identity_loss = identity_A_loss + identity_B_loss

      # Calcul la perte du générateur
      gen_AB_loss = self.generator_loss(d_fake_output_B)
      gen_BA_loss = self.generator_loss(d_fake_output_A)

      # Calcul perte cycle
      gen_AB_loss_2nd = self.generator_loss(d_fake_output_cycle_B)
      gen_BA_loss_2nd = self.generator_loss(d_fake_output_cycle_A)
  
      total_gen_loss = tf.dtypes.cast(gen_AB_loss, tf.float64) + tf.dtypes.cast(gen_BA_loss, tf.float64) + \
                       tf.dtypes.cast(gen_AB_loss_2nd, tf.float64) + tf.dtypes.cast(gen_BA_loss_2nd, tf.float64) + \
                       self.lambda_cycle * total_cycle_loss  + total_identity_loss * self.lambda_identity

      # Calcul des performances du générateur
      gen_AB_acc = self.generator_accuracy(d_fake_output_B)
      # Calcul des performances du générateur
      gen_BA_acc = self.generator_accuracy(d_fake_output_A)
      
    
    # Récupère les gradients des variables entraînables par rapport à la perte
    generator_vars = self.generator_AB.trainable_variables + self.generator_BA.trainable_variables
    gradients_of_generator = gen_tape.gradient(total_gen_loss, generator_vars)
    self.g_optimizer.apply_gradients(zip(gradients_of_generator, generator_vars))
    
    with tf.GradientTape() as dis_tape:
      # Classifie les vrais échantillons de la voix A
      real_output_A = self.discriminator_A(dataset_A_batch, training=True)
      # Classifie les vrais échantillons de la voix B
      real_output_B = self.discriminator_B(dataset_B_batch, training=True)
      
      generated_samples_A = self.generator_BA(dataset_B_batch, training=True)
      d_fake_output_A = self.discriminator_A(generated_samples_A, training=True)
      
      cycled_samples_B = self.generator_AB(tf.squeeze(generated_samples_A, axis=-1), training=True)
      d_cycled_B = self.discriminator_B(cycled_samples_B, training=True)
      
      generated_samples_B = self.generator_AB(dataset_A_batch, training=True)
      d_fake_output_B = self.discriminator_B(generated_samples_B, training=True)
      
      cycled_samples_A = self.generator_BA(tf.squeeze(generated_samples_B, axis=-1), training=True)
      d_cycled_A = self.discriminator_A(cycled_samples_A, training=True)
            
      # Calcul la perte du discriminateur
      disc_A_loss = self.discriminator_loss(real_output_A, d_fake_output_A)
      disc_B_loss = self.discriminator_loss(real_output_B, d_fake_output_B)
      
      # Calcul la perte du cycle
      disc_A_loss_2nd = self.discriminator_loss(real_output_A, d_cycled_A)
      disc_B_loss_2nd = self.discriminator_loss(real_output_B, d_cycled_B)
      
      total_disc_loss = (disc_A_loss + disc_B_loss) / 2.0 + (disc_A_loss_2nd + disc_B_loss_2nd) / 2.0

      # Calcul des performances du discriminateur
      disc_B_acc = self.discriminator_accuracy(real_output_B, d_fake_output_B)
      # Calcul des performances du discriminateur
      disc_A_acc = self.discriminator_accuracy(real_output_A, d_fake_output_A)
      # Calcul des performances du discriminateur avec les vrais échantillons de la voix B
      disc_B_real_acc = self.discriminator_real_accuracy(real_output_B)
      # Calcul des performances du discriminateur avec les vrais échantillons de la voix B
      disc_A_real_acc = self.discriminator_real_accuracy(real_output_A)
      # Calcul des performances du discriminateur avec les faux échantillons de la voix B
      disc_B_gen_acc = self.discriminator_generated_accuracy(d_fake_output_B)
      # Calcul des performances du discriminateur avec les faux échantillons de la voix B
      disc_A_gen_acc = self.discriminator_generated_accuracy(d_fake_output_A)

    
    discriminator_vars = self.discriminator_B.trainable_variables + self.discriminator_A.trainable_variables
    gradients_of_discriminator = dis_tape.gradient(total_disc_loss, discriminator_vars)
    self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_vars))

    return total_gen_loss, total_disc_loss, gen_AB_loss, gen_BA_loss, cycle_A_loss, cycle_B_loss, identity_B_loss, identity_A_loss, disc_B_loss, disc_A_loss, gen_AB_acc, gen_BA_acc, disc_B_acc, disc_B_real_acc, disc_B_gen_acc, disc_A_acc, disc_A_real_acc, disc_A_gen_acc

  def train(self, coded_sps_A_norm, coded_sps_B_norm):
    file_writer = tf.summary.create_file_writer(self.logdir)
    file_writer.set_as_default()

    time_list = []

    # Boucle sur les époques
    for epoch in range(self.epochs):
      start_time = time.time()
      coded_sps_A_norm_cut, coded_sps_B_norm_cut = utils.cut_coded_sps_norm(coded_sps_A_norm, coded_sps_B_norm, self.n_frames)
      dataset_A = tf.data.Dataset.from_tensor_slices(coded_sps_A_norm_cut)
      dataset_B = tf.data.Dataset.from_tensor_slices(coded_sps_B_norm_cut)

      dataset_A = dataset_A.shuffle(buffer_size=coded_sps_A_norm_cut.shape[0]).batch(self.batch_size)
      dataset_B = dataset_B.shuffle(buffer_size=coded_sps_B_norm_cut.shape[0]).batch(self.batch_size)

      # Boucle sur les lots du jeu de données
      for dataset_A_batch, dataset_B_batch in zip(dataset_A, dataset_B):
        num_iterations = coded_sps_A_norm_cut.shape[0] // self.batch_size * epoch

        if num_iterations > 10000:
          self.lambda_identity = 0
          self.g_learning_rate = max(0, self.g_learning_rate - self.g_learning_rate_decay)
          self.d_learning_rate = max(0, self.d_learning_rate - self.d_learning_rate_decay)
          self.g_optimizer.lr.assign(self.g_learning_rate)
          self.d_optimizer.lr.assign(self.d_learning_rate)
        # Entraine le GAN à l'aide d'un lot d'échantillons de la voix A et de la voix B
        total_gen_loss, total_disc_loss, gen_AB_loss, gen_BA_loss, cycle_A_loss, cycle_B_loss, identity_B_loss, identity_A_loss, disc_B_loss, disc_A_loss, gen_AB_acc, gen_BA_acc, disc_B_acc, disc_B_real_acc, disc_B_gen_acc, disc_A_acc, disc_A_real_acc, disc_A_gen_acc = self.train_step(dataset_A_batch, dataset_B_batch)
      

      wall_time_sec = time.time() - start_time
      time_list.append(wall_time_sec)
      
      if (epoch+1) % 20 == 0:
        print("Saving Epoch {}".format(epoch+1))
        ckpt_name = "generateurAB" + "-" + str(epoch+1)
        self.generator_AB.save(self.dir_model + ckpt_name)
        ckpt_name = "generateurBA" + "-" + str(epoch+1)
        self.generator_BA.save(self.dir_model + ckpt_name)

      template = 'Epoch {}, Generator loss {}, Discriminator loss {}, Generator AB loss {}, Generator BA loss {}, Cycle A loss {}, Cycle B loss {}, Identity B loss {}, Identity A loss {}, Discriminator B Loss {}, Discriminator A Loss {}, Generator AB accuracy {}, Generator BA accuracy {}, Discriminator B accuracy {}, Discriminator B real accuracy {}, Discriminator B fake accuracy {}, Discriminator A accuracy {}, Discriminator A real accuracy {}, Discriminator A fake accuracy {}'
      print (template.format(epoch, total_gen_loss, total_disc_loss, gen_AB_loss, gen_BA_loss, cycle_A_loss, cycle_B_loss, identity_B_loss, identity_A_loss, disc_B_loss, disc_A_loss, gen_AB_acc, gen_BA_acc, disc_B_acc, disc_B_real_acc, disc_B_gen_acc, disc_A_acc, disc_A_real_acc, disc_A_gen_acc))
      print("Iteration {}, Generateur LR {}, Discriminateur LR {}".format(num_iterations, self.g_learning_rate, self.d_learning_rate))

      # Logue les différentes pertes
      tf.summary.scalar('01. Generator loss', total_gen_loss, step=epoch)
      tf.summary.scalar('02. Discriminator loss', total_disc_loss, step=epoch)
      tf.summary.scalar('03. Generator AB loss', gen_AB_loss, step=epoch)
      tf.summary.scalar('04. Discriminator B loss', disc_B_loss, step=epoch)
      tf.summary.scalar('05. Generator BA loss', gen_BA_loss, step=epoch)
      tf.summary.scalar('06. Discriminator A loss', disc_A_loss, step=epoch)
      tf.summary.scalar('07. Cycle A loss', cycle_A_loss, step=epoch)
      tf.summary.scalar('08. Cycle B loss', cycle_B_loss, step=epoch)
      tf.summary.scalar('09. Identity B loss', identity_B_loss, step=epoch)
      tf.summary.scalar('10. Identity A loss', identity_A_loss, step=epoch)
      # Logue les différentes performances
      tf.summary.scalar('11. Generator accuracy AB', gen_AB_acc, step=epoch)
      tf.summary.scalar('12. Discriminator B accuracy', disc_B_acc, step=epoch)
      tf.summary.scalar('13. Discriminator B real voice', disc_B_real_acc, step=epoch)
      tf.summary.scalar('14. Discriminator B voice generated', disc_B_gen_acc, step=epoch)
      tf.summary.scalar('15. Generator accuracy BA', gen_AB_acc, step=epoch)
      tf.summary.scalar('16. Discriminator A accuracy', disc_A_acc, step=epoch)
      tf.summary.scalar('17. Discriminator A real voice', disc_A_real_acc, step=epoch)
      tf.summary.scalar('18. Discriminator A voice generated', disc_A_gen_acc, step=epoch)

      if (epoch) == 0 or (epoch+1) % 5 == 0:
        # Voix A->B
        filenames = list()
        filenames.append(os.listdir(self.dir_wavs_A_test)[0])
        filenames.append(random.choice(os.listdir(self.dir_wavs_A_test)))
        num = 1
        for file in filenames:
          wav, _ = librosa.load(os.path.join(self.dir_wavs_A_test, file), sr = self.sr, mono = True)
          wav = utils.wav_padding(wav, self.sr, self.frame_period, multiple = 4)
          f0, timeaxis, sp, ap = utils.world_decompose(wav, self.sr, self.frame_period)
          f0_converted = utils.pitch_conversion(f0, self.log_f0s_mean_A, self.log_f0s_std_A, self.log_f0s_mean_B, self.log_f0s_std_B)
          coded_sp = utils.world_encode_spectral_envelop(sp, self.sr, self.num_mcep) 
          coded_sp_transposed = coded_sp.T
          coded_sp_norm = utils.coded_sp_normalization_fit_transform(coded_sp_transposed, self.coded_sps_A_mean, self.coded_sps_A_std) 
          coded_sp_norm = np.array([coded_sp_norm])
          coded_sp_converted_norm = self.generator_AB(coded_sp_norm, training=False)
          coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
          coded_sp_converted = utils.coded_sp_denormalization_fit_transform(coded_sp_converted_norm, self.coded_sps_B_mean, self.coded_sps_B_std)
          coded_sp_converted = coded_sp_converted.T
          coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
          sp_converted = utils.world_decode_spectral_envelop(coded_sp_converted, self.sr)
          
          sp_original_decoded = utils.world_decode_spectral_envelop(coded_sp, self.sr)
          wav_original_decoded = utils.world_speech_synthesis(f0, sp_original_decoded, ap, self.sr, self.frame_period)
          
          wav_transformed = utils.world_speech_synthesis(f0_converted, sp_converted, ap, self.sr, self.frame_period)

          tf.summary.audio(f'0{num}. Original_A_{file.rsplit(".")[-2]}', tf.expand_dims(tf.expand_dims(wav_original_decoded, -1), 0),
                            sample_rate=self.sr, step=epoch)
          tf.summary.audio(f'0{num+1}. Transformed_A->B_{file.rsplit(".")[-2]}', tf.expand_dims(tf.expand_dims(wav_transformed, -1), 0),
                            sample_rate=self.sr, step=epoch)

          fig = pyplot.figure(figsize=(16,5))
          librosa.display.specshow(np.log(sp_original_decoded).T,
                                    sr=self.sr,
                                    hop_length=int(0.001 * self.sr * self.frame_period),
                                    x_axis="time",
                                    y_axis="linear",
                                    cmap="magma")
          pyplot.colorbar()
          tf.summary.image(f'0{num}. Original A', plot_to_image(fig), step=epoch)

          fig = pyplot.figure(figsize=(16,5))
          librosa.display.specshow(np.log(sp_converted).T,
                                    sr=self.sr,
                                    hop_length=int(0.001 * self.sr * self.frame_period),
                                    x_axis="time",
                                    y_axis="linear",
                                    cmap="magma")
          pyplot.colorbar()
          tf.summary.image(f'0{num+1}. Transformed A-B', plot_to_image(fig), step=epoch)
          num = num + 2

        # Voix B->A
        filenames = list()
        filenames.append(os.listdir(self.dir_wavs_B_test)[0])
        filenames.append(random.choice(os.listdir(self.dir_wavs_B_test)))
        for file in filenames:
          wav, _ = librosa.load(os.path.join(self.dir_wavs_B_test, file), sr = self.sr, mono = True)
          wav = utils.wav_padding(wav, self.sr, self.frame_period, multiple = 4)
          f0, timeaxis, sp, ap = utils.world_decompose(wav, self.sr, self.frame_period)
          f0_converted = utils.pitch_conversion(f0, self.log_f0s_mean_B, self.log_f0s_std_B, self.log_f0s_mean_A, self.log_f0s_std_A)
          coded_sp = utils.world_encode_spectral_envelop(sp, self.sr, self.num_mcep) 
          coded_sp_transposed = coded_sp.T
          coded_sp_norm = utils.coded_sp_normalization_fit_transform(coded_sp_transposed, self.coded_sps_B_mean, self.coded_sps_B_std) 
          coded_sp_norm = np.array([coded_sp_norm])
          coded_sp_converted_norm = self.generator_BA(coded_sp_norm, training=False)
          coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
          coded_sp_converted = utils.coded_sp_denormalization_fit_transform(coded_sp_converted_norm, self.coded_sps_A_mean, self.coded_sps_A_std)
          coded_sp_converted = coded_sp_converted.T
          coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
          sp_converted = utils.world_decode_spectral_envelop(coded_sp_converted, self.sr)
          
          sp_original_decoded = utils.world_decode_spectral_envelop(coded_sp, self.sr)
          wav_original_decoded = utils.world_speech_synthesis(f0, sp_original_decoded, ap, self.sr, self.frame_period)
          
          wav_transformed = utils.world_speech_synthesis(f0_converted, sp_converted, ap, self.sr, self.frame_period)

          tf.summary.audio(f'0{num}. Original_B_{file.rsplit(".")[-2]}', tf.expand_dims(tf.expand_dims(wav_original_decoded, -1), 0),
                            sample_rate=self.sr, step=epoch)
          tf.summary.audio(f'0{num+1}. Transformed_B->A_{file.rsplit(".")[-2]}', tf.expand_dims(tf.expand_dims(wav_transformed, -1), 0),
                            sample_rate=self.sr, step=epoch)

          fig = pyplot.figure(figsize=(16,5))
          librosa.display.specshow(np.log(sp_original_decoded).T,
                                    sr=self.sr,
                                    hop_length=int(0.001 * self.sr * self.frame_period),
                                    x_axis="time",
                                    y_axis="linear",
                                    cmap="magma")
          pyplot.colorbar()
          tf.summary.image(f'0{num}. Original B', plot_to_image(fig), step=epoch)

          fig = pyplot.figure(figsize=(16,5))
          librosa.display.specshow(np.log(sp_converted).T,
                                    sr=self.sr,
                                    hop_length=int(0.001 * self.sr * self.frame_period),
                                    x_axis="time",
                                    y_axis="linear",
                                    cmap="magma")
          pyplot.colorbar()
          tf.summary.image(f'0{num+1}. Transformed B->A', plot_to_image(fig), step=epoch)
          num = num + 2

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