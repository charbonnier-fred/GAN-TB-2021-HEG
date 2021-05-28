import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pyworld
import soundfile as sf

# Charge les fichiers wav dans une liste
# wav_dir : Répertoire
# sr : Taux d'échantillonnage
def load_wavs(wav_dir, sr):
  wavs = list()
  filenames = list()
  for file in os.listdir(wav_dir):
    file_path = os.path.join(wav_dir, file)
    wav, _ = librosa.load(file_path, sr=sr, mono=True)
    wavs.append(wav)
    filenames.append(file)
  return wavs, filenames

# Extrait la f0, l'enveloppe spectrale et l'apériodicité
# d'un fichier audio à l'aide de WORLD vocodeur
# wav : Fichier audio
# fs : Taux d'échantillonnage
# frame_period : Durée d'une trame
def world_decompose(wav, fs, frame_period=5.0):
  wav = wav.astype(np.float64)
  # Extraction de la f0 et de la position temporelle de chaque trame
  f0, timeaxis = pyworld.harvest(
      wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)

  # Extraction de l'enveloppe spectrale
  sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)

  # Extraction de l'aperiodicité
  ap = pyworld.d4c(wav, f0, timeaxis, fs)

  return f0, timeaxis, sp, ap

# Génère la représentation Mel-cepstral coefficients (MCEP)
# sp : Enveloppe spectrale
# fs : Taux d'échantillonnage
# dim : Nombre de dimensions MCEP souhaitées
def world_encode_spectral_envelop(sp, fs, dim=34):
  # Get Mel-Cepstral coefficients (MCEP)
  coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
  return coded_sp

def world_encode_data(wave, fs, frame_period=5.0, coded_dim=34):
  f0s = list()
  timeaxes = list()
  sps = list()
  aps = list()
  coded_sps = list()
  for wav in wave:
      f0, timeaxis, sp, ap = world_decompose(wav=wav,
                                              fs=fs,
                                              frame_period=frame_period)
      coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
      f0s.append(f0)
      timeaxes.append(timeaxis)
      sps.append(sp)
      aps.append(ap)
      coded_sps.append(coded_sp)
  return f0s, timeaxes, sps, aps, coded_sps

# Calcule la moyenne et l'écart type de la
# fréquence fondamentale
def logf0_statistics(f0s):
  log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
  log_f0s_mean = log_f0s_concatenated.mean()
  log_f0s_std = log_f0s_concatenated.std()
  return log_f0s_mean, log_f0s_std

# Transpose dans une liste
def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst

# Normalise la représentation Mel-cepstral coefficients (MCEP)
def coded_sps_normalization_fit_transform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append(coded_sp_normalization_fit_transform(coded_sp, coded_sps_mean, coded_sps_std))
    return coded_sps_normalized, coded_sps_mean, coded_sps_std

def coded_sp_normalization_fit_transform(coded_sp, coded_sps_mean, coded_sps_std):
    return (coded_sp - coded_sps_mean) / coded_sps_std

# Dénormalise les représentations Mel-cepstral coefficients (MCEP)
def coded_sps_denormalization_fit_transform(coded_sps_normalized, coded_sps_mean, coded_sps_std):
    coded_sps = list()
    for coded_sp_normalized in coded_sps_normalized:
      coded_sps.append(coded_sp_denormalization_fit_transform(coded_sp_normalized, coded_sps_mean, coded_sps_std))
    return coded_sps

# Dénormalise la représentation Mel-cepstral coefficients (MCEP)
def coded_sp_denormalization_fit_transform(coded_sp_normalized, coded_sps_mean, coded_sps_std):
    return coded_sp_normalized * coded_sps_std + coded_sps_mean

# Décode l'enveloppe spectrale
# Mel-cepstral coefficients (MCEP) -> envelope spectrale
def world_decode_spectral_envelop(coded_sp, fs):
  fftlen = pyworld.get_cheaptrick_fft_size(fs)
  decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
  return decoded_sp

def world_decode_spectral_envelops(coded_sps, fs):
  decoded_sps = list()
  for coded_sp in coded_sps:
      decoded_sps.append(world_decode_spectral_envelop(coded_sp, fs))
  return decoded_sps

# Génère des fichiers audios depuis leurs
# caractéristiques
# f0 : Fréquence fondamentale
# decoded_sp : Enveloppe spectrale décodée
# ap : apériodicité
# fs : Taux d'échantillonnage
# frame_period : Durée d'une trame 
def world_speechs_synthesis(f0s, decoded_sps, aps, fs, frame_period):
  wavs = list()
  for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
    wavs.append(world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period))
  return wavs

def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
  wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
  return wav.astype(np.float32)

# Sauvegarde les fichiers audios
# output_dir : Répertoire de sortie
# file_basenames : Noms des fichiers
# wavs : liste de wavs
def save_wavs(output_dir, filenames, wavs, sr):
  for file_basename, wav in zip(filenames, wavs):
    sf.write(os.path.join(output_dir, file_basename), wav, sr, 'PCM_24')

# Normalisation gaussienne logarithmique pour la conversion du pitch
def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
  f0_converted = np.exp((np.log(f0) - mean_log_src) /
                        std_log_src * std_log_target + mean_log_target)
  return f0_converted

def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)

def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def preprocessing(dir_wavs_A, dir_wavs_B, dir_cache, sr, frame_period, num_mcep,
                  path_logf0s_norm, path_mcep_norm, path_coded_sps_A_norm, path_coded_sps_B_norm):
  # Vérifie que le cache soit vide
  if not os.path.exists(dir_cache):
    os.mkdir(dir_cache)
    # Chargement des fichiers wav des voix A et B
    wavs_A, _ = load_wavs(dir_wavs_A, sr)
    wavs_B, _ = load_wavs(dir_wavs_B, sr)

    # Extraction des caractéristiques des voix A et B
    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs_A, sr, frame_period, num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs_B, sr, frame_period, num_mcep)

    # Calcul de la moyenne et de l'écart type du f0 des voix A et B
    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    # Transposition MCEP des voix A et B
    coded_sps_A_transposed = transpose_in_list(coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(coded_sps_B)

    # Normalisation MCEP des voix A et B
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transform(coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transform(coded_sps_B_transposed)

    np.savez(path_logf0s_norm,
              mean_A=log_f0s_mean_A,
              std_A=log_f0s_std_A,
              mean_B=log_f0s_mean_B,
              std_B=log_f0s_std_B)

    np.savez(path_mcep_norm,
              mean_A=coded_sps_A_mean,
              std_A=coded_sps_A_std,
              mean_B=coded_sps_B_mean,
              std_B=coded_sps_B_std)
    
    save_pickle(variable=coded_sps_A_norm,
                  fileName=path_coded_sps_A_norm)
    save_pickle(variable=coded_sps_B_norm,
                fileName=path_coded_sps_B_norm)
  # Si le cache n'est pas vide
  else:
    coded_sps_A_norm = load_pickle_file(path_coded_sps_A_norm)
    coded_sps_B_norm = load_pickle_file(path_coded_sps_B_norm)
  
  return coded_sps_A_norm, coded_sps_B_norm

# Découpe les MCEP normalisés à un nombre de frames défini 
def cut_coded_sps_norm(coded_sps_A_norm, coded_sps_B_norm, n_frames=128):
  coded_sps_A_norm_split = list()
  coded_sps_B_norm_split = list()

  for coded_sp_A_norm, coded_sp_B_norm in zip(coded_sps_A_norm, coded_sps_B_norm):
      frames_A_total = coded_sp_A_norm.shape[1]
      assert frames_A_total >= n_frames
      start_A = np.random.randint(frames_A_total - n_frames + 1)
      end_A = start_A + n_frames
      coded_sps_A_norm_split.append(coded_sp_A_norm[:, start_A:end_A])

      frames_B_total = coded_sp_B_norm.shape[1]
      assert frames_B_total >= n_frames
      start_B = np.random.randint(frames_B_total - n_frames + 1)
      end_B = start_B + n_frames
      coded_sps_B_norm_split.append(coded_sp_B_norm[:, start_B:end_B])

  coded_sps_A_norm_split = np.array(coded_sps_A_norm_split)
  coded_sps_B_norm_split = np.array(coded_sps_B_norm_split)

  return coded_sps_A_norm_split, coded_sps_B_norm_split

#Formate le fichier audio pour être compatible avec l'entrée du générateur
def wav_padding(wav, sr, frame_period, multiple = 4):
  assert wav.ndim == 1 
  num_frames = len(wav)
  num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) 
                          * (sr * frame_period / 1000))
  num_frames_diff = num_frames_padded - num_frames
  num_pad_left = num_frames_diff // 2
  num_pad_right = num_frames_diff - num_pad_left
  wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

  return wav_padded

# Visualiser la forme d'onde
def wp_show(wave, fs, title):
  plt.figure(figsize=(16,5))
  librosa.display.waveplot(wave, sr=fs)
  plt.title(title)
  plt.show()

# Visualiser la fréquence fondamentale
def f0_show(f0, timeaxe, title):
  plt.figure(figsize=(16,5))
  plt.plot(timeaxe, f0)
  plt.title(title)
  plt.xlabel('Time')
  plt.ylabel('Hz')
  plt.xlim(0, timeaxe[-1])
  plt.show()

# Visualiser l'enveloppe spectrale
def sp_show(sp, fs, frame_period, title):
  plt.figure(figsize=(16,5))
  librosa.display.specshow(np.log(sp).T,
                           sr=fs,
                           hop_length=int(0.001 * fs * frame_period),
                           x_axis="time",
                           y_axis="linear",
                           cmap="magma")
  plt.title(title)
  plt.colorbar()
  plt.show()

# Visualiser l'enveloppe spectrale encodée (MCEP)
def coded_sp_show(coded_sp, fs, frame_period, title):
  plt.figure(figsize=(16,5))
  librosa.display.specshow(coded_sp.T,
                           sr=fs,
                           hop_length=int(0.001 * fs * frame_period),
                           x_axis="time",
                           cmap="magma")
  plt.ylabel('MCEP')
  plt.title(title)
  plt.colorbar()
  plt.show()
