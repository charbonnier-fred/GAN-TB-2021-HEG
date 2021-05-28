import keyboard
import librosa
import numpy as np
import os
import pyaudio
import soundfile as sf
import tensorflow as tf
import utils
import wave

class Recorder(object):

  def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
    self.channels = channels
    self.rate = rate
    self.frames_per_buffer = frames_per_buffer

  def open(self, fname, mode='wb'):
    return RecordingFile(fname, mode, self.channels, self.rate,
              self.frames_per_buffer)

class RecordingFile(object):
  def __init__(self, fname, mode, channels, 
        rate, frames_per_buffer):
    self.fname = fname
    self.mode = mode
    self.channels = channels
    self.rate = rate
    self.frames_per_buffer = frames_per_buffer
    self._pa = pyaudio.PyAudio()
    self.wavefile = self._prepare_file(self.fname, self.mode)
    self._stream = None

  def __enter__(self):
    return self

  def __exit__(self, exception, value, traceback):
    self.close()

  def record(self, duration):
    # Use a stream with no callback function in blocking mode
    self._stream = self._pa.open(format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.frames_per_buffer)
    for _ in range(int(self.rate / self.frames_per_buffer * duration)):
      audio = self._stream.read(self.frames_per_buffer)
      self.wavefile.writeframes(audio)
    return None

  def start_recording(self):
    # Use a stream with a callback in non-blocking mode
    self._stream = self._pa.open(format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.frames_per_buffer,
                    stream_callback=self.get_callback())
    self._stream.start_stream()
    return self

  def stop_recording(self):
    self._stream.stop_stream()
    return self

  def get_callback(self):
    def callback(in_data, frame_count, time_info, status):
      self.wavefile.writeframes(in_data)
      return in_data, pyaudio.paContinue
    return callback


  def close(self):
    self._stream.close()
    self._pa.terminate()
    self.wavefile.close()

  def _prepare_file(self, fname, mode='wb'):
    wavefile = wave.open(fname, mode)
    wavefile.setnchannels(self.channels)
    wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
    wavefile.setframerate(self.rate)
    return wavefile

dir_cache = "cache/"
path_logf0s_norm = os.path.join(dir_cache, 'logf0s_normalization.npz')
path_mcep_norm = os.path.join(dir_cache, 'mcep_normalization.npz')
logf0s_normalization = np.load(path_logf0s_norm)
log_f0s_mean_A = logf0s_normalization['mean_A']
log_f0s_std_A = logf0s_normalization['std_A']
log_f0s_mean_B = logf0s_normalization['mean_B']
log_f0s_std_B = logf0s_normalization['std_B']
mcep_normalization = np.load(path_mcep_norm)
coded_sps_A_mean = mcep_normalization['mean_A']
coded_sps_A_std = mcep_normalization['std_A']
coded_sps_B_mean = mcep_normalization['mean_B']
coded_sps_B_std = mcep_normalization['std_B']
temp_wav_file = "temp.wav"
temp_converted_wav_file ="tempconvert.wav"
sr = 22000
frame_period = 5.0
num_mcep = 36
n_frames = 128

generator_AB = tf.keras.models.load_model('model/generateurAB-820')
rec = Recorder()

while True:
  print("Ready?")
  keyboard.wait(' ')
  
  with rec.open('temp.wav', 'wb') as recfile2:
    print("Speak....")
    recfile2.start_recording()
    keyboard.wait(' ')
    recfile2.stop_recording()
  print("Conversion...")
  wav, _ = librosa.load(temp_wav_file, sr = sr, mono = True)
  wav = utils.wav_padding(wav, sr, frame_period, multiple = 4)
  f0, timeaxis, sp, ap = utils.world_decompose(wav, sr, frame_period)
  f0_converted = utils.pitch_conversion(f0, log_f0s_mean_A, log_f0s_std_A, log_f0s_mean_B, log_f0s_std_B)
  coded_sp = utils.world_encode_spectral_envelop(sp, sr, num_mcep) 
  coded_sp_transposed = coded_sp.T
  coded_sp_norm = utils.coded_sp_normalization_fit_transform(coded_sp_transposed, coded_sps_A_mean, coded_sps_A_std) 
  coded_sp_norm = np.array([coded_sp_norm])
  coded_sp_converted_norm = generator_AB.predict(coded_sp_norm)
  coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
  coded_sp_converted = utils.coded_sp_denormalization_fit_transform(coded_sp_converted_norm, coded_sps_B_mean, coded_sps_B_std)
  coded_sp_converted = coded_sp_converted.T
  coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
  sp_converted = utils.world_decode_spectral_envelop(coded_sp_converted, sr)

  wav_transformed = utils.world_speech_synthesis(f0_converted, sp_converted, ap, sr, frame_period)
  sf.write(temp_converted_wav_file, wav_transformed, sr, 'PCM_24')

  # Set chunk size of 1024 samples per data frame
  chunk = 1024  

  # Open the sound file 
  wf = wave.open(temp_converted_wav_file, 'rb')

  # Create an interface to PortAudio 
  p = pyaudio.PyAudio()

  # Open a .Stream object to write the WAV file to
  # 'output = True' indicates that the sound will be played rather than recorded
  stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                  channels = wf.getnchannels(),
                  rate = wf.getframerate(),
                  output = True)

  # Read data in chunks
  data = wf.readframes(chunk)

  print("Play....")
  # Play the sound by writing the audio data to the stream
  while data :
    stream.write(data)
    data = wf.readframes(chunk)

  # Close and terminate the stream
  stream.close()
  p.terminate()