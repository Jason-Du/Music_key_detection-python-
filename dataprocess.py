import librosa
import ffmpeg
import numpy as np
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import time
import multiprocessing as mp
from multiprocessing import Process, Pool
from  scipy import signal
class Feature():
  def __init__(self):
    Feature.Chroma_stft = np.array([])
    Feature.Chroma_cens=np.array([])
    Feature.Chroma_cqt=np.array([])
    Feature.FlatnessX=np.array([])
    Feature.ContrastX=np.array([])
    Feature.SpecbwX=np.array([])
    Feature.RmseX=np.array([])
    Feature.MelspecX=np.array([])
    Feature.TonnetzX=np.array([])
    Feature.MfccX=np.array([])
    Feature.CentroidX=np.array([])
    Feature.Zero_crossX=np.array([])
    Feature.RolloffX=np.array([])
    Feature.OenvX = np.array([])
    Feature.TempoX = np.array([])
    Feature.DtempoX = np.array([])
    Feature.Ac_globalX = np.array([])
    Feature.Tempogram = np.array([])


class Make_Train_Data():
  def __init__(self, music_dir_name, cut_music_dir_name, npy_file_dir_name, music_split_dir, cut_music_split_dir, npy_split_dir):
    self.capitalname=music_dir_name
    self.music_path=os.path.join(os.path.dirname(__file__), music_dir_name)
    self.cut_music_path=os.path.join(os.path.dirname(__file__),cut_music_dir_name)
    self.npy_path=os.path.join(os.path.dirname(__file__),npy_file_dir_name)
    self.split_capitalname=music_split_dir
    self.split_musicpath=os.path.join(os.path.dirname(__file__),music_split_dir)
    self.split_npypath=os.path.join(os.path.dirname(__file__),npy_split_dir)
    self.spit_cutmusicpath=os.path.join(os.path.dirname(__file__), cut_music_split_dir)
    super().__init__()
  def division_music(self):
    alldirList=os.listdir(self.music_path)
    for dir in alldirList:
      if dir=="MinorA":
        print("ALL Major is done")
        print("Major cut file num:{}".format(total_file_num(input_dirname_path=self.cut_music_path)))
      songarray = []
      allfileList = os.listdir(os.path.join(self.music_path,dir))
      for file in allfileList:
        file = file.strip(".mp3")
        songarray.append(file)
      for song_index in range(len(songarray)):
        filename = songarray[song_index]
        filenameinput = filename + ".mp3"
        file_path = os.path.join(self.music_path,dir,filenameinput)
        if os.path.isfile(file_path):
          print("{} is exist".format(filename))
          # AudioSegment.converter = r"D:\\ffmpeg\\ffmpeg-20200814-a762fd2-win64-static\\bin"
          audio = AudioSegment.from_file(file_path, format="mp3")
          # print(audio.duration_seconds)
          decard_sec_front = 0
          audio_sec_num = int((audio.duration_seconds - decard_sec_front) / 30)
          # print(audio_sec_num)
          for i in range(audio_sec_num):
            filenameexport = filename + str(i) + ".mp3"
            audio[(i * 30 * 1000) + decard_sec_front * 1000:(i + 1) * 30 * 1000 + decard_sec_front * 1000].export(
            os.path.join(self.cut_music_path,dir,filenameexport))
              # os.path.join(os.path.dirname(__file__), "7Tone/MajorA_cut", filenameexport))
          print("{} is doen".format(filename))
        else:
          print("ERROR {} ; {} file is not exist".format(dir,filename))
      print("{} is done".format(dir))

  def feature_accerlate(self,featurename):
    alldirList = os.listdir(self.cut_music_path)
    chroma_stft_list = []
    chroma_cqt_list = []
    chroma_cens_list = []
    flatness_list = []
    contrast_list = []
    specbw_list = []
    rmse_list = []
    melspec_list = []
    tonnetz_list = []
    mfcc_list = []
    centroid_list = []
    zero_crossing_rate_list = []
    rolloff_list = []
    hop_length = 512
    tempo_list = []
    tempogram_list = []
    oenv_list = []
    acglobal_list = []
    dtempo_list = []
    # READ DIRdir_NAME
    dir_index = 0
    for dir in alldirList:
      major_branch_path = os.path.join(self.cut_music_path, dir)
      major_branch_file_num, major_branch_file_array = dir_file_num(major_branch_path)
      for song_index in range(major_branch_file_num):
        print("------------------------------------------")
        filename = major_branch_file_array[song_index]
        file_path = os.path.join(major_branch_path, filename)

        y, sr = librosa.load(file_path, sr=10000)
        # b, a = signal.butter(N=8, Wn=0.88, btype="low", analog=False)  # 配置濾波器 8 表示濾波器的階數
        # y = signal.filtfilt(b, a, y)  # data為要過濾的訊號

        if featurename=="flatness":
          pass
          flatness = librosa.feature.spectral_flatness(y=y)
          flatness = flatness[np.newaxis, :]
          flatness_list.append(flatness)
        if featurename=="contrast":
          pass
          s = librosa.stft(y)
          S = np.abs(s)
          contrast = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=80)
          contrast = contrast[np.newaxis, :]
          contrast_list.append(contrast)
        if featurename=="specbw":
          pass
          specbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
          specbw = specbw[np.newaxis, :]
          specbw_list.append(specbw)
        if featurename=="rmse":
          pass
          rmse = librosa.feature.rmse(y=y)
          rmse = rmse[np.newaxis, :]
          rmse_list.append(rmse)
        if featurename=="tonnetz":
          H = librosa.effects.harmonic(y)
          tonnetz = librosa.feature.tonnetz(y=H, sr=sr)
          tonnetz = tonnetz[np.newaxis, :]
          tonnetz_list.append(tonnetz)
          pass
        if featurename=="mfcc":
          mfcc = librosa.feature.mfcc(y, sr=sr)
          mfcc = mfcc[np.newaxis, :]
          mfcc_list.append(mfcc)
          pass
        if featurename=="centroid":
          centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
          centroid = centroid[np.newaxis, :]
          centroid_list.append(centroid)
          pass
        if featurename=="zero_crossing_rate":
          zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
          zero_crossing_rate = zero_crossing_rate[np.newaxis, :]
          zero_crossing_rate_list.append(zero_crossing_rate)
          pass
        if featurename=="rolloff":
          rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
          rolloff = rolloff[np.newaxis, :]
          rolloff_list.append(rolloff)
          pass
        if featurename=="melspec":
          melspec = librosa.feature.melspectrogram(y=y, sr=sr)
          melspec = melspec[np.newaxis, :]
          melspec_list.append(melspec)
          pass
        if featurename=="chroma_stft":
          chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
          chroma_stft=chroma_stft[np.newaxis,:]
          chroma_stft_list.append(chroma_stft)
          pass
        if featurename=="chroma_cens":
          chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
          chroma_cens=chroma_cens[np.newaxis,:]
          chroma_cens_list.append(chroma_cens)
          pass
        if featurename=="chroma_cqt":
          chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
          chroma_cqt=chroma_cqt[np.newaxis,:]
          chroma_cqt_list.append(chroma_cqt)
          pass
        if featurename=="oenv":
          oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
          onev = onev[np.newaxis, :]
          oenv_list.append(oenv)
        if featurename=="tempogram":
          oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
          tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
          tempogram= tempogram[np.newaxis, :]
          tempogram_list.append(tempogram)
        # Compute global onset autocorrelation
        # 384
        if featurename == "ac_global":
          oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
          tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
          ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
          ac_global=ac_global[np.newaxis, :]
          acglobal_list.append(ac_global)
        # Estimate the global tempo for display purposes
        # 1
        if featurename == "tempo":
          oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
          tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)
          tempo=tempo[np.newaxis,:]
          tempo_list.append(tempo)
          # static
        # 1319
        if featurename == "dtempo":
          dtempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length, aggregate=None)
          dtempo = dtempo[np.newaxis, :]
          dtempo_list.append(dtempo)
        print("{} {} {} is done".format(featurename,dir,filename))
      print("******************************************")
      print("{} directory is done".format(dir))
      print("******************************************")
      dir_index += 1
    if featurename == "flatness":
      Feature.FlatnessX = np.vstack((flatness_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Flatness_Datasetx"), Feature.FlatnessX)
    if featurename == "contrast":
      Feature.ContrastX = np.vstack((contrast_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Contrast_Datasetx"), Feature.ContrastX)
    if featurename == "specbw":
      Feature.SpecbwX = np.vstack((specbw_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Specbw_Datasetx"), Feature.SpecbwX)
    if featurename == "rmse":
      Feature.RmseX = np.vstack((rmse_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Rmse_Datasetx"), Feature.RmseX)
    if featurename == "tonnetz":
      Feature.TonnetzX = np.vstack((tonnetz_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Tonnetz_Datasetx"), Feature.TonnetzX)
    if featurename == "mfcc":
      Feature.MfccX = np.vstack((mfcc_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_MFCC_Datasetx"), Feature.MfccX)
    if featurename == "centroid":
      Feature.CentroidX = np.vstack((centroid_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Centroid_Datasetx"), Feature.CentroidX)
    if featurename == "zero_crossing_rate":
      Feature.Zero_crossX = np.vstack((zero_crossing_rate_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Zero_cross_Datasetx"), Feature.Zero_crossX)
    if featurename == "rolloff":
      Feature.RolloffX = np.vstack((rolloff_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Rolloff_Datasetx"), Feature.RolloffX)
    if featurename == "melspec":
      Feature.MelspecX = np.vstack((melspec_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Melspec_Datasetx"), Feature.MelspecX)
    if featurename == "chroma_stft":
      Feature.Chroma_stftX = np.vstack((chroma_stft_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Chroma_stft_Datasetx"), Feature.Chroma_stftX)
    if featurename == "chroma_cens":
      Feature.Chroma_censX = np.vstack((chroma_cens_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Chroma_cens_Datasetx"), Feature.Chroma_censX)
    if featurename == "chroma_cqt":
      Feature.Chroma_cqtX = np.vstack((chroma_cqt_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Chroma_cqt_Datasetx"), Feature.Chroma_cqtX)
    if featurename == "oenv":
      Feature.OenvX=np.vstack((oenv_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Oenv_Datasetx"), Feature.OenvX)
    if featurename == "tempogram":
      Feature.Tempogram=np.vstack((tempogram_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Tempogram_Datasetx"), Feature.Tempogram)
    # Compute global onset autocorrelation
    # 384
    if featurename == "ac_global":
      Feature.Ac_global=np.vstack((acglobal_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Ac_global_Datasetx"), Feature.Ac_global)
    # Estimate the global tempo for display purposes
    # 1
    if featurename == "tempo":
      Feature.TempoX=np.vstack((tempo_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Tempo_Datasetx"), Feature.TempoX)  # static
    # 1319
    if featurename == "dtempo":
      Feature.DtempoX = np.vstack((dtempo_list))
      np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Dtempo_Datasetx"), Feature.DtempoX)
  def feature(self):
    alldirList = os.listdir(self.cut_music_path)
    # READ DIRdir_NAME
    dir_index = 0
    for dir in alldirList:
      major_branch_path = os.path.join(self.cut_music_path, dir)
      major_branch_file_num, major_branch_file_array = dir_file_num(major_branch_path)
      for song_index in range(major_branch_file_num):
        print("------------------------------------------")
        filename = major_branch_file_array[song_index]
        file_path = os.path.join(major_branch_path, filename)

        y, sr = librosa.load(file_path, sr=10000)
        H = librosa.effects.harmonic(y)
        s = librosa.stft(y)
        S = np.abs(s)
        flatness = librosa.feature.spectral_flatness(y=y)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=80)
        specbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        tonnetz = librosa.feature.tonnetz(y=H, sr=sr)
        mfcc = librosa.feature.mfcc(y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        melspec = librosa.feature.melspectrogram(y=y, sr=sr)
        chroma_stft=librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cens=librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)


        print("{} is done".format(filename))
        chroma_stft=chroma_stft[np.newaxis,:]
        chroma_cens=chroma_cens[np.newaxis,:]
        chroma_cqt=chroma_cqt[np.newaxis,:]
        flatness = flatness[np.newaxis, :]
        contrast = contrast[np.newaxis, :]
        specbw = specbw[np.newaxis, :]
        rmse = rmse[np.newaxis, :]
        melspec = melspec[np.newaxis, :]
        tonnetz = tonnetz[np.newaxis, :]
        mfcc = mfcc[np.newaxis, :]
        centroid = centroid[np.newaxis, :]
        zero_crossing_rate = zero_crossing_rate[np.newaxis, :]
        rolloff = rolloff[np.newaxis, :]
        if (song_index == 0 and dir_index == 0):
          Feature.Chroma_stftX = chroma_stft
          Feature.Chroma_censX =chroma_cens
          Feature.Chroma_cqtX = chroma_cqt
          Feature.FlatnessX = flatness
          Feature.ContrastX = contrast
          Feature.SpecbwX = specbw
          Feature.RmseX = rmse
          Feature.MelspecX = melspec
          Feature.TonnetzX = tonnetz
          Feature.MfccX = mfcc
          Feature.CentroidX = centroid
          Feature.Zero_crossX = zero_crossing_rate
          Feature.RolloffX = rolloff
          print("Chroma_stftX shape:{}".format(Feature.Chroma_stftX.shape))
          print("Chroma_censX shape:{}".format(Feature.Chroma_censX.shape))
          print("Chroma_cqtX shape:{}".format(Feature.Chroma_cqtX.shape))
          print("FlatnessX shape:{}".format(Feature.FlatnessX.shape))
          print("ContrastX shape:{}".format(Feature.ContrastX.shape))
          print("SpecbwX shape:{}".format(Feature.SpecbwX.shape))
          print("RmseX shape:{}".format(Feature.RmseX.shape))
          print("MelspecX shape:{}".format(Feature.MelspecX.shape))
          print("TonnetzX shape:{}".format(Feature.TonnetzX.shape))
          print("MfccX shape:{}".format(Feature.MfccX.shape))
          print("CentroidX shape:{}".format(Feature.CentroidX.shape))
          print("Zero_crossX shape:{}".format(Feature.Zero_crossX.shape))
          print("RolloffX shape:{}".format(Feature.RolloffX.shape))

          continue
        Feature.Chroma_stftX=np.vstack((Feature.Chroma_stftX,chroma_stft))
        Feature.Chroma_censX=np.vstack((Feature.Chroma_censX,chroma_cens))
        Feature.Chroma_cqtX=np.vstack((Feature.Chroma_cqtX, chroma_cqt))
        Feature.FlatnessX = np.vstack((Feature.FlatnessX,flatness))
        Feature.ContrastX = np.vstack(( Feature.ContrastX,contrast))
        Feature.SpecbwX = np.vstack((Feature.SpecbwX,specbw))
        Feature.RmseX = np.vstack(( Feature.RmseX,rmse))
        Feature.MelspecX = np.vstack((Feature.MelspecX,melspec))
        Feature.TonnetzX = np.vstack(( Feature.TonnetzX,tonnetz))
        Feature.MfccX = np.vstack(( Feature.MfccX,mfcc))
        Feature.CentroidX = np.vstack(( Feature.CentroidX,centroid))
        Feature.Zero_crossX = np.vstack((Feature.Zero_crossX,zero_crossing_rate))
        Feature.RolloffX = np.vstack(( Feature.RolloffX,rolloff))
        print("Chroma_stftX shape:{}".format(Feature.Chroma_stftX.shape))
        print("Chroma_censX shape:{}".format(Feature.Chroma_censX.shape))
        print("Chroma_cqtx shape:{}".format(Feature.Chroma_cqtX.shape))
        print("FlatnessX shape:{}".format(Feature.FlatnessX.shape))
        print("ContrastX shape:{}".format(Feature.ContrastX.shape))
        print("SpecbwX shape:{}".format(Feature.SpecbwX.shape))
        print("RmseX shape:{}".format(Feature.RmseX.shape))
        print("MelspecX shape:{}".format(Feature.MelspecX.shape))
        print("TonnetzX shape:{}".format(Feature.TonnetzX.shape))
        print("MfccX shape:{}".format(Feature.MfccX.shape))
        print("CentroidX shape:{}".format(Feature.CentroidX.shape))
        print("Zero_crossX shape:{}".format(Feature.Zero_crossX.shape))
        print("RolloffX shape:{}".format(Feature.RolloffX.shape))

      print("******************************************")
      print("{} directory is done".format(dir))
      print("******************************************")
      dir_index += 1
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Chroma_cqt_Datasetx"), Feature.Chroma_cqtX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Chroma_cens_Datasetx"), Feature.Chroma_censX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Chroma_stft_Datasetx"), Feature.Chroma_stftX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Flatness_Datasetx"), Feature.FlatnessX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Contrast_Datasetx"), Feature.ContrastX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Specbw_Datasetx"), Feature.SpecbwX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Rmse_Datasetx"), Feature.RmseX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Melspec_Datasetx"), Feature.MelspecX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Tonnetz_Datasetx"), Feature.TonnetzX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_MFCC_Datasetx"), Feature.MfccX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Centroid_Datasetx"), Feature. CentroidX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Zero_cross_Datasetx"), Feature.Zero_crossX)
    np.save(os.path.join(self.npy_path, "Tone24_cut30_Rolloff_Datasetx"), Feature.RolloffX)

  def rhythm(self):
    alldirList = os.listdir(self.cut_music_path)
    tempo_list=[]
    tempogram_list = []
    oenv_list=[]
    acglobal_list=[]
    dtempo_list=[]
    # READ DIRdir_NAME
    dir_index = 0
    for dir in alldirList:
      major_branch_path = os.path.join(self.cut_music_path, dir)
      major_branch_file_num, major_branch_file_array = dir_file_num(major_branch_path)
      for song_index in range(major_branch_file_num):

        print("------------------------------------------")
        filename = major_branch_file_array[song_index]
        file_path = os.path.join(major_branch_path, filename)

        y, sr = librosa.load(file_path, sr=22500)
        b, a = signal.butter(N=8, Wn=0.88, btype="low", analog=False)  # 配置濾波器 8 表示濾波器的階數
        y = signal.filtfilt(b, a, y)  # data為要過濾的訊號
        hop_length = 512
        # 1319
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        # 384,1319
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        # Compute global onset autocorrelation
        # 384
        ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
        # Estimate the global tempo for display purposes
        # 1
        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)  # static
        # 1319
        dtempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length, aggregate=None)
        oenv=oenv[np.newaxis,...]
        tempogram=tempogram[np.newaxis,...]
        ac_global=ac_global[np.newaxis,...]
        tempo=tempo[np.newaxis,...]
        dtempo=dtempo[np.newaxis,...]
        print("{} {} is done".format(dir, filename))
        oenv_list.append(oenv)
        tempogram_list.append(tempogram)
        acglobal_list.append(ac_global)
        tempo_list.append(tempo)
        dtempo_list.append(dtempo)
        # debug
        # if song_index==1:
        #   break

      print("******************************************")
      print("{} directory is done".format(dir))
      print("******************************************")
      dir_index += 1
    Feature.OenvX = np.vstack((oenv_list))
    Feature.TempoX = np.vstack((tempo_list))
    Feature.DtempoX = np.vstack((dtempo_list))
    Feature.Ac_globalX = np.vstack((acglobal_list))
    Feature.Tempogram = np.vstack((tempogram_list))
    np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Oenv_Datasetx"), Feature.OenvX)
    np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Tempo_Datasetx"), Feature.TempoX)
    np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Dtempo_Datasetx"), Feature.DtempoX)
    np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Ac_global_Datasetx"), Feature.Ac_globalX)
    np.save(os.path.join(self.npy_path, self.capitalname + "_cut30_Tempogram_Datasetx"), Feature.Tempogram)

  def normalize(self,feature_data_name):
    feature_data_name_input = feature_data_name + ".npy"
    feature_data = np.load(os.path.join(self.npy_path, feature_data_name_input))
    if feature_data_name=="24Tone_cut30_Tempogram_Datasetx":
      feature_data=feature_data[:,0:100,:]
    if feature_data_name == "24Tone_cut30_Melspec_Datasetx_log":
      feature_data = feature_data[:, 0:128, :]
    data_num = feature_data.shape[0]
    feature_data_list=[]

    for data_index in range(data_num):
      feature_single_data = feature_data[data_index, ...]
      feature_single_data = sklearn.preprocessing.maxabs_scale(feature_single_data, axis=0)
      feature_single_data_after=feature_single_data[np.newaxis,:]
      feature_data_list.append(feature_single_data_after)
      print("Normalizing {}feature_data_list NUM:{}".format(feature_data_name,data_index))

    feature_normalize_data = np.vstack((feature_data_list))
    print(feature_data_name," feature_normalize_data shape:{}".format(feature_normalize_data.shape))
    feature_output_data_name = feature_data_name + "_N"
    np.save(os.path.join(self.npy_path, feature_output_data_name), feature_normalize_data)
    print(feature_data_name, "feature_normalize_data shape:{}".format(feature_normalize_data.shape))

  def Dataset(self):
    Trainy = make_trainy_data(input_dirname_path=self.cut_music_path)
    print("Trainy shpae:{}".format(Trainy.shape))
    print("Tone24_cut30_Datasety:{}".format(Trainy))
    np.save(os.path.join(self.npy_path, self.capitalname+"_cut30_Datasety"), Trainy)
    # ******************************************************************************************
    # 12
    # Feature.Chroma_stft = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Chroma_stft_Datasetx.npy"))
    # Feature.Chroma_cens = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Chroma_cens_Datasetx.npy"))
    Feature.Chroma_cqt = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Chroma_cqt_Datasetx.npy"))
    # 20
    Feature.MfccX = np.load(os.path.join(self.npy_path,self.capitalname+"_cut30_Mfcc_Datasetx_N.npy"))
    # 1
    # Feature.CentroidX = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Centroid_Datasetx.npy"))
    # Feature.Zero_crossX = np.load(os.path.join(self.npy_path,self.capitalname+"_cut30_Zero_cross_Datasetx.npy"))
    # Feature.RolloffX = np.load(os.path.join(self.npy_path,self.capitalname+"_cut30_Rolloff_Datasetx.npy"))
    # 6
    Feature.TonnetzX = np.load(os.path.join(self.npy_path,self.capitalname+"_cut30_Tonnetz_Datasetx_N.npy"))
    # 1
    # Feature.FlatnessX = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Flatness_Datasetx.npy"))
    # # 7
    # Feature.ContrastX = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Contrast_Datasetx_N.npy"))
    # 1
    # Feature.SpecbwX = np.load(os.path.join(self.npy_path,self.capitalname+"_cut30_Specbw_Datasetx.npy"))
    # Feature.RmseX = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Rmse_Datasetx.npy"))
    # 128
    Feature.MelspecX = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Melspec_Datasetx_log_N.npy"))
    # 7365 1 586
    # Feature.OenvX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Oenv_Datasetx_N.npy"))
    # Feature.OenvX = Feature.OenvX[:, np.newaxis, :]
    # # 7365 1
    # Feature.TempoX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Tempo_Datasetx_N.npy"))
    # # *********************7365 1 586
    # Feature.DtempoX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Dtempo_Datasetx_N.npy"))
    # Feature.DtempoX = Feature.DtempoX[:, np.newaxis, :]
    # # 7365 384
    # Feature.Ac_globalX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Ac_global_Datasetx_N.npy"))
    # # ***********************7365 384 586
    # Feature.Tempogram = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Tempogram_Datasetx_N.npy"))
    # DatasetX=np.stack((Feature.MelspecX,Feature.Tempogram),axis=3)
    DatasetX = np.hstack((Feature.Chroma_cqt,Feature.MfccX,Feature.TonnetzX,Feature.MelspecX))
    # DatasetX=np.hstack((Feature.MelspecX,Feature.Tempogram))
    # DatasetX=Feature.MelspecX
    print("DatasetX shape:{}".format(DatasetX.shape))
    np.save(os.path.join(self.npy_path, self.capitalname+"_cut30_Datasetx.npy"), DatasetX)

  def make_train_vaild_test(self):
    Datasetx = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Datasetx.npy"))
    Datasety = np.load(os.path.join(self.npy_path, self.capitalname+"_cut30_Datasety.npy"))

    Trainx, Trainy, Vaildx, Vaildy, Testx, Testy = make_vaild_data_test_data_shuffel_train_data(
      dataset_x=Datasetx,
      dataset_y=Datasety,
      vaild_data_percentage=0.0001,
      test_data_percentage=0.0001
    )
    print("Trainx.shape:{}".format(Trainx.shape))
    print("Trainy.shape:{}".format(Trainy.shape))
    print("Vaildx.shape:{}".format(Vaildx.shape))
    print("Vaildy.shape:{}".format(Vaildy.shape))
    print("Testx.shape:{}".format(Testx.shape))
    print("Testy.shape:{}".format(Testy.shape))

    np.save(os.path.join(self.npy_path, "Trainx"), Trainx)
    np.save(os.path.join(self.npy_path, "Trainy"), Trainy)
    np.save(os.path.join(self.npy_path, "Vaildx"), Vaildx)
    np.save(os.path.join(self.npy_path, "Vaildy"), Vaildy)
    np.save(os.path.join(self.npy_path, "Testx"), Testx)
    np.save(os.path.join(self.npy_path, "Testy"), Testy)

  def make_log_spec(self,feature_data_name):
    feature_log_list=[]
    feature_data_name_input=feature_data_name+".npy"
    feature_data_name_output=feature_data_name+"_log.npy"
    feature=np.load(os.path.join(os.path.dirname(__file__),self.npy_path,feature_data_name_input))
    for index in range(feature.shape[0]):
      feature_log=-1*(librosa.power_to_db(feature[index,...],ref=np.max))
      feature_log_after=feature_log[np.newaxis,...]
      feature_log_list.append(feature_log_after)
      print("{}file num:{}".format(feature_data_name,index))

    feature_log_out=np.vstack((feature_log_list))
    print(feature_data_name, "shpape :{}".format(feature_log_out.shape))
    np.save(os.path.join(os.path.dirname(__file__),self.npy_path,feature_data_name_output),feature_log_out)
    pass

  def make_split_feature(self):
    print("split featuring")
   #  Feature.Chroma_stft = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Chroma_stft_Datasetx.npy"))
   #  Feature.Chroma_cens = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Chroma_cens_Datasetx.npy"))
    Feature.Chroma_cqt = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Chroma_cqt_Datasetx.npy"))
   #  # 20
    Feature.MfccX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Mfcc_Datasetx_N.npy"))
   #  # 1
   #  Feature.CentroidX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Centroid_Datasetx_N.npy"))
   #  Feature.Zero_crossX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Zero_cross_Datasetx_N.npy"))
   #  Feature.RolloffX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Rolloff_Datasetx_N.npy"))
   # # 6
    Feature.TonnetzX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Tonnetz_Datasetx_N.npy"))
   #  # 1
   #  Feature.FlatnessX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Flatness_Datasetx_N.npy"))
   #  # 7
   #  Feature.ContrastX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Contrast_Datasetx_N.npy"))
   #  # 1
   #  Feature.SpecbwX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Specbw_Datasetx_N.npy"))
   #  Feature.RmseX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Rmse_Datasetx_N.npy"))
   #  # 128
    Feature.MelspecX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Melspec_Datasetx_log_N.npy"))
   #  # 7365 ********************586
   #  Feature.OenvX =np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Oenv_Datasetx_N.npy"))
   #  Feature.OenvX=Feature.OenvX[:,np.newaxis,:]
   #  # 7365 1
   #  Feature.TempoX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Tempo_Datasetx_N.npy"))
   #  # *********************7365 586
   #  Feature.DtempoX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Dtempo_Datasetx_N.npy"))
   #  Feature.DtempoX=Feature.DtempoX[:,np.newaxis,:]
   #  7365 384
   #  Feature.Ac_globalX = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Ac_global_Datasetx_N.npy"))
   #  # ***********************7365 384 586
   #  Feature.Tempogram = np.load(os.path.join(self.npy_path, self.capitalname + "_cut30_Tempogram_Datasetx_N.npy"))
    if self.split_capitalname=="12MajorTone":
      pass
      # Feature.Chroma_stft=Feature.Chroma_stft[0:4301,:,:]
      # Feature.Chroma_cens=Feature.Chroma_cens[0:4301,:,:]
      # Feature.Chroma_cqt=Feature.Chroma_cqt[0:4301,:,:]
      # Feature.MfccX=Feature.MfccX[0:4301,:,:]
      # Feature.CentroidX=Feature.CentroidX[0:4301,:,:]
      # Feature.Zero_crossX=Feature.Zero_crossX[0:4301,:,:]
      # Feature.RolloffX=Feature.RolloffX[0:4301,:,:]
      # Feature.TonnetzX = Feature.TonnetzX[0:4301,:,:]
      # Feature.FlatnessX = Feature.FlatnessX[0:4301,:,:]
      # Feature.ContrastX = Feature.ContrastX [0:4301,:,:]
      # Feature.SpecbwX = Feature.SpecbwX[0:4301,:,:]
      # Feature.RmseX = Feature.RmseX[0:4301,:,:]
      Feature.MelspecX=Feature.MelspecX[0:4301,:,:]
      # Feature.OenvX = Feature.OenvX[0:4301,:]
      # Feature.TempoX =  Feature.TempoX[0:4301,:]
      # Feature.DtempoX = Feature.DtempoX[0:4301,:]
      # Feature.Ac_globalX = Feature.Ac_globalX[0:4301,:]
      # Feature.Tempogram = Feature.Tempogram[0:4301,:,:]
    elif self.split_capitalname=="12MinorTone":
      pass
      # Feature.Chroma_stft=Feature.Chroma_stft[4301:,:,:]
      # Feature.Chroma_cens=Feature.Chroma_cens[4301:,:,:]
      # Feature.Chroma_cqt=Feature.Chroma_cqt[4301:,:,:]
      # Feature.MfccX=Feature.MfccX[4301:,:,:]
      # Feature.CentroidX=Feature.CentroidX[4301:,:,:]
      # Feature.Zero_crossX=Feature.Zero_crossX[4301:,:,:]
      # Feature.RolloffX=Feature.RolloffX[4301:,:,:]
      # Feature.TonnetzX = Feature.TonnetzX[4301:,:,:]
      # Feature.FlatnessX = Feature.FlatnessX[4301:,:,:]
      # Feature.ContrastX = Feature.ContrastX [4301:,:,:]
      # Feature.SpecbwX = Feature.SpecbwX[4301:,:,:]
      # Feature.RmseX = Feature.RmseX[4301:,:,:]
      # Feature.MelspecX=Feature.MelspecX[4301:,:,:]
      # Feature.OenvX = Feature.OenvX[4301:,:]
      # Feature.TempoX =  Feature.TempoX[4301:,:]
      # Feature.DtempoX = Feature.DtempoX[4301:,:]
      # Feature.Ac_globalX = Feature.Ac_globalX[4301:,:]
      # Feature.Tempogram = Feature.Tempogram[4301:,:,:]


    np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Chroma_cqt_Datasetx"), Feature.Chroma_cqt)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Chroma_cens_Datasetx"), Feature.Chroma_cens)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Chroma_stft_Datasetx"), Feature.Chroma_stft)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Flatness_Datasetx"), Feature.FlatnessX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Contrast_Datasetx"), Feature.ContrastX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Specbw_Datasetx"), Feature.SpecbwX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Rmse_Datasetx"), Feature.RmseX)
    np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Melspec_Datasetx"), Feature.MelspecX)
    np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Tonnetz_Datasetx"), Feature.TonnetzX)
    np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_MFCC_Datasetx"), Feature.MfccX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Centroid_Datasetx"), Feature. CentroidX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Zero_cross_Datasetx"), Feature.Zero_crossX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Rolloff_Datasetx"), Feature.RolloffX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Tempo_Datasetx"), Feature.TempoX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Dtempo_Datasetx"), Feature.DtempoX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Tempogram_Datasetx"), Feature. Tempogram)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Ac_global_Datasetx"), Feature.Ac_globalX)
    # np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Oenv_Datasetx"), Feature.OenvX)
    print("split featuring finish")

  def make_split_dataset(self):
    pass
    Trainy = make_trainy_data(input_dirname_path=self.spit_cutmusicpath)
    print("Trainy shpae:{}".format(Trainy.shape))
    print("Tone24_cut30_Datasety:{}".format(Trainy))
    np.save(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Datasety"), Trainy)
    # ******************************************************************************************
    # 12
    # Feature.Chroma_stft = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Chroma_stft_Datasetx.npy"))
    # Feature.Chroma_cens = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Chroma_cens_Datasetx.npy"))
    Feature.Chroma_cqt = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Chroma_cqt_Datasetx.npy"))
    # # 20
    Feature.MfccX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Mfcc_Datasetx.npy"))
    # # 1
    # Feature.CentroidX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Centroid_Datasetx.npy"))
    # Feature.Zero_crossX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Zero_cross_Datasetx.npy"))
    # Feature.RolloffX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Rolloff_Datasetx.npy"))
    # # 6
    Feature.TonnetzX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Tonnetz_Datasetx.npy"))
    # # 1
    # Feature.FlatnessX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Flatness_Datasetx.npy"))
    # # 7
    # Feature.ContrastX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Contrast_Datasetx.npy"))
    # # 1
    # Feature.SpecbwX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Specbw_Datasetx.npy"))
    # Feature.RmseX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Rmse_Datasetx.npy"))
    # # 128
    Feature.MelspecX = np.load(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Melspec_Datasetx.npy"))
    #
    # Feature.TempoX= np.load(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Tempo_Datasetx.npy"))
    # # 7365 1 586
    # Feature.DtempoX= np.load(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Dtempo_Datasetx.npy"))
    # # 7365 384 586
    # Feature. Tempogram= np.load(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Tempogram_Datasetx.npy"))
    #
    # Feature.Ac_globalX= np.load(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Ac_global_Datasetx.npy"))
    # # 7365,1 586
    # Feature.OenvX= np.load(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Oenv_Datasetx.npy"))

    # DatasetX=np.hstack((Feature.Chroma_cqt))
    DatasetX = np.hstack((Feature.Chroma_cqt,Feature.MfccX,Feature.TonnetzX,Feature.MelspecX))
    # DatasetX = Feature.MelspecX
    print("DatasetX shape:{}".format(DatasetX.shape))
    np.save(os.path.join(self.split_npypath, self.split_capitalname + "_cut30_Datasetx.npy"), DatasetX)

  def make_split_train_vaild_test(self):
    Datasetx = np.load(os.path.join(self.split_npypath, self.split_capitalname+"_cut30_Datasetx.npy"))
    Datasety = np.load(os.path.join(self.split_npypath,self.split_capitalname+"_cut30_Datasety.npy"))
    if self.split_capitalname=="2Tone":
      # FIX
      Datasetx=Datasetx[996:1619,...]
    print("DatasetX shape:{}".format(Datasetx.shape))

    Trainx, Trainy, Vaildx, Vaildy, Testx, Testy = make_vaild_data_test_data_shuffel_train_data(
      dataset_x=Datasetx,
      dataset_y=Datasety,
      vaild_data_percentage=0.01,
      test_data_percentage=0.01
    )
    print("Trainx.shape:{}".format(Trainx.shape))
    print("Trainy.shape:{}".format(Trainy.shape))
    print("Vaildx.shape:{}".format(Vaildx.shape))
    print("Vaildy.shape:{}".format(Vaildy.shape))
    print("Testx.shape:{}".format(Testx.shape))
    print("Testy.shape:{}".format(Testy.shape))

    np.save(os.path.join(self.split_npypath, "Trainx"), Trainx)
    np.save(os.path.join(self.split_npypath, "Trainy"), Trainy)
    np.save(os.path.join(self.split_npypath, "Vaildx"), Vaildx)
    np.save(os.path.join(self.split_npypath, "Vaildy"), Vaildy)
    np.save(os.path.join(self.split_npypath, "Testx"), Testx)
    np.save(os.path.join(self.split_npypath, "Testy"), Testy)



def get_shuffle_Data(dataset_X, dataset_Y):
  arrayRandomIndex = np.arange(len(dataset_X))
  np.random.shuffle(arrayRandomIndex)
  return dataset_X[arrayRandomIndex], dataset_Y[arrayRandomIndex]

def make_vaild_data_test_data_shuffel_train_data(dataset_x,dataset_y,vaild_data_percentage,test_data_percentage):
  intInputDataSize =dataset_x.shape[0]
  intValidDataSize=int(np.floor(intInputDataSize * vaild_data_percentage))
  intTestDataSize = int(np.floor(intInputDataSize * test_data_percentage))
  intTrainDataSize=intInputDataSize-intValidDataSize-intTestDataSize
  dataset_x_shuffel,dataset_y_shuffel=get_shuffle_Data(dataset_X=dataset_x,dataset_Y=dataset_y)
  Trainx=dataset_x_shuffel[0:intTrainDataSize]
  Trainy=dataset_y_shuffel[0:intTrainDataSize]
  Vaildx=dataset_x_shuffel[intTrainDataSize:intTrainDataSize+intValidDataSize]
  Vaildy=dataset_y_shuffel[intTrainDataSize:intTrainDataSize+intValidDataSize]
  Testx=dataset_x_shuffel[intTrainDataSize+intValidDataSize:]
  Testy=dataset_y_shuffel[intTrainDataSize+intValidDataSize:]
  return Trainx,Trainy,Vaildx,Vaildy,Testx,Testy
# TrainYDATA
def make_trainy_data(input_dirname_path):
  alldirList = os.listdir(input_dirname_path)
  trainy=np.array([])
  index=0
  for dir in alldirList:
    single_dir_file_num,array=dir_file_num(os.path.join(input_dirname_path, dir))
    for i in range(single_dir_file_num):
      trainy=np.append(trainy,index)
    index +=1
    print(index)
  return trainy




#計算DATA數
# return songarray
def dir_file_num(input_dirname_path):
  songarray = []
  allfileList = os.listdir(input_dirname_path)
  for file in allfileList:
    songarray.append(file)
  return len(songarray),songarray
# doublelayer
def total_file_num(input_dirname_path):
  alldirList = os.listdir(input_dirname_path)
  total_dir_file_num=0.0
  for dir in alldirList:
    single_dir_file_num,array=dir_file_num(os.path.join(input_dirname_path,dir))
    total_dir_file_num=total_dir_file_num+single_dir_file_num
  return total_dir_file_num


if __name__ == '__main__':

    tt = time.time()
    Tone24_Major=Make_Train_Data(music_dir_name="24Tone",
                               cut_music_dir_name="24Tone_cut",
                               npy_file_dir_name="24Tone_cut_npy_file",
                               music_split_dir="12MajorTone",
                               cut_music_split_dir="12MajorTone_cut",
                               npy_split_dir="12MajorTone_cut_npy_file",
                               )
    Tone24_Minor = Make_Train_Data(music_dir_name="24Tone",
                                 cut_music_dir_name="24Tone_cut",
                                 npy_file_dir_name="24Tone_cut_npy_file",
                                 music_split_dir="12MinorTone",
                                 cut_music_split_dir="12MinorTone_cut",
                                 npy_split_dir="12MinorTone_cut_npy_file",
                                 )
    Tone24_SVM = Make_Train_Data(music_dir_name="24Tone",
                               cut_music_dir_name="24Tone_cut",
                               npy_file_dir_name="24Tone_cut_npy_file",
                               music_split_dir="SVMTone",
                               cut_music_split_dir="SVMTone_cut",
                               npy_split_dir="SVMTone_cut_npy_file",
                               )
    Tone2_SVM= Make_Train_Data(music_dir_name="24Tone",
                               cut_music_dir_name="24Tone_cut",
                               npy_file_dir_name="24Tone_cut_npy_file",
                               music_split_dir="2Tone",
                               cut_music_split_dir="2Tone_cut",
                               npy_split_dir="2Tone_cut_npy_file",
                               )
    Mm12=Make_Train_Data(music_dir_name="Mm12Tone",
                               cut_music_dir_name="Mm12Tone_cut",
                               npy_file_dir_name="Mm12Tone_cut_npy_file",
                               music_split_dir="2Tone",
                               cut_music_split_dir="2Tone_cut",
                               npy_split_dir="2Tone_cut_npy_file",

    )
    # #     # 12大小調具群 製造feature
    # pool = Pool(4)
    # processlist = ["tempogram",
    #                 "mfcc"
    #                 ]
    # pool.map( Mm12.feature_accerlate, processlist)
    # pool.close()
    # pool.join()
    # Mm12.make_log_spec(feature_data_name="Mm12Tone_cut30_Melspec_Datasetx")
    # Mm12.normalize(feature_data_name="Mm12Tone_cut30_MFCC_Datasetx")
    # pool=Pool(4)
    # processlist=["Mm12Tone_cut30_Melspec_Datasetx_log",
    #           "Mm12Tone_cut30_Tonnetz_Datasetx",
    #           "Mm12Tone_cut30_Contrast_Datasetx",
    #           ]
    # pool.map( Mm12.normalize, processlist)
    # pool.close()
    # pool.join()
    #

    # Tone24.division_music()
    # Tone24.feature()
    # Tone24_Major.rhythm()
    # pool=Pool(12)
    # processlist=["24Tone_cut30_Tonnetz_Datasetx",
    #              "24Tone_cut30_MFCC_Datasetx",
    #              "24Tone_cut30_Flatness_Datasetx",
    #              "24Tone_cut30_Rmse_Datasetx",
    #              "24Tone_cut30_Rolloff_Datasetx",
    #              "24Tone_cut30_Centroid_Datasetx",
    #              "24Tone_cut30_Contrast_Datasetx",
    #              "24Tone_cut30_Specbw_Datasetx",
    #              "24Tone_cut30_Zero_cross_Datasetx"
    #              ]
    # pool.map(Tone24_Major.normalize, processlist)
    # pool.close()
    # pool.join()
    #
    # Tone24_Major.make_log_spec("24Tone_cut30_Melspec_Datasetx")
    # Tone24_Major.normalize("24Tone_cut30_Melspec_Datasetx_log")
    # pool = Pool(12)
    #     # processlist=["24Tone_cut30_Oenv_Datasetx",
    #     # 		   "24Tone_cut30_Tempo_Datasetx",
    #     # 		   "24Tone_cut30_Dtempo_Datasetx",
    #     # 		   "24Tone_cut30_Ac_global_Datasetx",
    #     # 		   "24Tone_cut30_Tempogram_Datasetx",
    #     # 		   ]
    #     # pool.map(Tone24_Major.normalize, processlist)
    #     # pool.close()
    #     # pool.join()
    # 產生24首分類
    # Tone24_Major.normalize("24Tone_cut30_Tempogram_Datasetx")
    Tone24_Major.Dataset()
    Tone24_Major.make_train_vaild_test()

    # 產生大調12首分類
    # Tone24_Major.make_split_feature()
    # Tone24_Major.make_split_dataset()
    # Tone24_Major.make_split_train_vaild_test()
    # 產生小調12首分類
    # Tone24_Minor.make_split_feature()
    # Tone24_Minor.make_split_dataset()
    # Tone24_Minor.make_split_train_vaild_test()

    # 產生大小調分2類
    # Tone24_SVM.make_split_feature()
    # Tone24_SVM.make_split_dataset()
    # Tone24_SVM.make_split_train_vaild_test()

    # 小集群分辨大小調
    # Mm12.make_split_feature()
    # Mm12.make_split_dataset()
    # Mm12.make_split_train_vaild_test()
    # 12大小調具群Mm
    # Mm12.Dataset()
    # Mm12.make_train_vaild_test()


    print('Time used: {} sec'.format(time.time()-tt))