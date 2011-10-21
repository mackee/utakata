# -*- coding:utf8 -*-
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.linalg as slng

import time
def stopwatch(wrapped):
  """計時デコレータ"""
  def _wrapper(*args, **kwargs):
    tic = time.time()
    result = wrapped(*args, **kwargs)
    toc = time.time()
    doc = str(wrapped.__doc__).split("\n")[0]
    print("[%s] %f[sec]" % (doc, toc - tic))
    return result
  return _wrapper


#信号処理ハンドラ
class BaseProcessHandler:
  """Base Class for Signal Process"""
  prevHandler = []

  def __init__(self, prevHandler):
    self.prevHandler = prevHandler

  def __getattr__(self, name):
    """ 自分が持っていないプロパティの場合、このメソッドが自動的に
        呼ばれて移譲する"""
    return getattr(self.prevHandler, name)


class PlotTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - グラフにプロットする"""
  def __init__(self, prevHandler, source_name='time_freq'):
    BaseProcessHandler.__init__(self, prevHandler)
    source = getattr(self, source_name)
    self.plot(source)

  def plot(self, source):
    try:
      self.figure += 1
    except AttributeError:
      self.figure = 1

    extent = [0, self.duration, 0, sp.shape(source)[0]]
    plt.figure(self.figure)
    plt.ylabel('key')
    plt.xlabel('time[sec]')
    plt.imshow(source, aspect='auto', origin='lower', extent=extent)


class MultipleMatfileAndTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - MATLABファイルの行列との積をとる"""
  def __init__(self, prevHandler, matrix_file, pinv=False, maximum=None):
    BaseProcessHandler.__init__(self, prevHandler)
    self.importMatrixFile(matrix_file)
    if(pinv):
      self.load_matrix = slng.pinv(self.load_matrix)
    self.multiple(maximum)

  def importMatrixFile(self, matrix_file):
    self.load_matrix = sio.loadmat(matrix_file)[matrix_file]

  def multiple(self, maximum=None):
    self.time_freq = sp.dot(self.load_matrix, self.time_freq)
    if(maximum != None):
      self.time_freq = sp.maximum(self.time_freq, maximum)


class BinarizeTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 二値化をする"""
  def __init__(self, prevHandler, threshold, target_data='time_freq'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.binarize(getattr(self, target_data), threshold)

  def binarize(self, target_data, threshold):
    data = sp.maximum(target_data, threshold)
    sp.putmask(data, data>threshold, 1.)
    sp.putmask(data, data<=threshold, 0.)
    self.binarized_data = data


class NormalizeTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 正規化をする"""
  def __init__(
      self, prevHandler, target='time_freq',
      set_name='time_freq', window=None):
    BaseProcessHandler.__init__(self, prevHandler)
    setattr(
        self, set_name,
        self.normalize(getattr(self, target), window))

  def normalize(self, target,  window):
    if(window == None):
      return self.time_freq / self.time_freq.max()
    
    else:
      if(window <= 1):
        fs = sp.shape(target)[1] / self.duration
        window_data = fs * window
      else:
        window_data = window
      
      normalized = sp.copy(target)
      for i in range(0, sp.shape(normalized)[1], int(window_data)):
        max_value = normalized[:, i:i+window_data].max()
        normalized[:, i:i+window_data] /= max_value
      
      return normalized
 

class LogTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 対数を取る"""
  def __init__(self, prevHandler):
    BaseProcessHandler.__init__(self, prevHandler)
    self.log()

  def log(self):
    self.time_freq = sp.log10(self.time_freq)

class CutOffNoiseTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - テンポ情報をもとにノイズ除去"""
  def __init__(self, prevHandler, minnotel=1./4.):
    BaseProcessHandler.__init__(self, prevHandler)
    self.scanSound(minnotel)

  def scanSound(self, minnotel):
    binarized = self.binarized_data
    scale = 60. / self.wavetempo * (binarized[0].size / self.duration)
    noise_length = scale*minnotel

    antinoised = sp.zeros_like(binarized)

    for i in range(sp.shape(binarized)[0]):
      new_line = binarized[i, :].copy()
      diffed = sp.diff(new_line)
      ones_keys = sp.where(diffed == 1)[0]
      minus_keys = sp.where(diffed == -1)[0]
      
      if(ones_keys.size != 0 and minus_keys.size != 0):
        if(ones_keys[0] > minus_keys[0]):
          new_line = self.cutNoise(
              (0, minus_keys[0]), noise_length, new_line)
          minus_keys = sp.delete(minus_keys, 0)

        if(ones_keys[-1] > minus_keys[-1]):
          new_line = self.cutNoise(
              (ones_keys[-1], new_line.size-1), noise_length, new_line)
          ones_keys = sp.delete(ones_keys, -1)

        for j in range(sp.size(ones_keys)):
          new_line = self.cutNoise(
              (ones_keys[j], minus_keys[j]), noise_length, new_line)

        antinoised[i, :] = new_line

    self.antinoised = antinoised


  def cutNoise(self, keys, noise_length, line):
    if(keys[1] - keys[0] < noise_length):
      array_ranges = sp.arange(keys[0], keys[1]+1)
      line[array_ranges] = sp.zeros_like(array_ranges)
    else:
      pass
    return line


class GradOnPitchTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 音程が高くなるにつれて強調する"""
  def __init__(self, prevHandler, factor, coef):
    BaseProcessHandler.__init__(self, prevHandler)
    self.gradTimeFreq(factor, coef)

  def gradTimeFreq(self, factor, coef):
    tf = self.time_freq
    time_freq = [tf[i, :]*(i**(1./factor))*(coef/(i+1)) for i in range(sp.shape(tf)[0])]
    self.time_freq = sp.array(time_freq)


class GenerateScoreTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - MMLデータを生成"""
  def __init__(
      self, prevHandler, target='time_freq',
      cut_num=40, output_form='MML'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.output_form = output_form

    self.left_score = self.generateScore(
        getattr(self, target)[:cut_num, :], cut_num)
    self.key_offset = cut_num
    self.prev_key = []
    self.right_score = self.generateScore(
        getattr(self, target)[cut_num:, :], cut_num)
    try:
      if(output_form == 'MML'):
        print self.convertMML(self.left_score)
        print self.convertMML(self.right_score)
      elif(output_form == 'PMX'):
        print self.convertPMX(self.left_score)
        print self.convertPMX(self.right_score)
    except TypeError:
      pass

  def generateScore(self, target, cut_num):
    fs = self.time_freq_fs = sp.shape(target)[1]  / self.duration
    noise_length = 60. / self.wavetempo * fs
    working = sp.copy(target)
    score = []
    while(sp.shape(working)[1] > noise_length):
      cutted, working = self.cutOutToChangingPoint(working)
      note = self.extractNote(cutted, cut_num)
      score.append(note)
    return score

  def cutOutToChangingPoint(self, working):
    timedomain = sp.array(
        [sp.sum(working[:, i]) for i in range(sp.shape(working)[1])])
    init_data = timedomain[0]
    try:
      changing_point = sp.where(timedomain != init_data)[0][0]
    except IndexError:
      changing_point = -1
    cutted = working[:, :changing_point]
    working = working[:, changing_point:]
    
    try:
      for i in self.prev_key:
        cutted[i, :] = sp.zeros_like(cutted[i, :])
    except AttributeError:
      pass

    return cutted, working

  def extractNote(self, cutted, cut_num):
    phonetic = self.calcPhonetic(sp.shape(cutted)[1])
    chord = self.extractChord(cutted)
    return (chord, phonetic)

  def calcPhonetic(self, length):
    note_sample = self.generateNoteLength()
    for name, length_pair in note_sample:
      if(length_pair[0] <= length and length < length_pair[1]):
        return name

  def extractChord(self, cutted):
    key_list = sp.where(cutted[:, 0] == 1)
    chord = self.convertToScale(key_list[0])
    return chord

  def convertToScale(self, key_list):
    try:
      key_offset = self.key_offset
    except AttributeError:
      key_offset = 0

    self.prev_key = key_list
    note_scales = self.generateNoteScales()
    chord = [note_scales[i+key_offset] for i in key_list]
    return chord
      
  def generateNoteLength(self):
    length = (60. / self.wavetempo) * self.time_freq_fs
    note_length = sp.array([2**i for i in range(5)]) / 4.
    note_length *= length
    note_huten = sp.array(
        [note_length[i-1]+note_length[i] for i in range(1, 5)])
    note_length = sp.r_[note_length, note_huten]
    note_length = sp.sort(note_length)

    note_length_pair = []
    for i in range(note_length.size):
      try:
        upper = (note_length[i+1] - note_length[i])/2
        upper += note_length[i]
      except IndexError:
        upper = note_length[i] * 2
      try:
        lower = note_length_pair[-1][1]
      except IndexError:
        lower = 0
      note_length_pair.append((lower, upper))
        
    if(self.output_form == 'MML'):
      note_name = ['16', '16.', '8', '8.', '4', '4.', '2', '2.', '1']
    elif(self.output_form == 'PMX'):
      note_name = ['1', '1d', '8', '8d', '4', '4d', '2', '2d', '0']
    return zip(note_name, note_length_pair)

  def generateNoteScales(self):
    if(self.output_form == 'MML'):
      scales = ['a', 'a+', 'b', 'c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+']
    elif(self.output_form == 'PMX'):
      scales = ['a', 'as', 'b', 'c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs']
    scales = (scales * 9)[:88]
    octaves = []
    for i in range(9):
      octaves += ([i]*12)
    octaves = octaves[10:98]
    return zip(scales, octaves)
 
  def convertMML(self, score):
    mml = ''
    now_octave = 3

    for note in score:
      if(note[0] == []):
        mml += 'r' + note[1]
      else:
        for scale in note[0]:
          if(scale[1] < now_octave):
            mml += '<' * (now_octave-scale[1])
            now_octave = scale[1]
          elif(now_octave < scale[1]):
            mml += '>' * (scale[1]-now_octave)
            now_octave = scale[1]
          mml += scale[0] + '0'
        mml = mml[:-1] + note[1]
    return mml

  def convertPMX(self, score):
    pmx = ''

    for note in score:
      if(note[0] == []):
        pmx += 'r' + note[1] + ' '
      else:
        length = note[1]
        for scale in note[0]:
          pmx += scale[0] + length + str(scale[1]) + ' z'
          length = ''
        pmx = pmx[:-2] + ' '
    return pmx


class EstimateTempoTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - テンポを推定"""
  def __init__(
    self, prevHandler, source='time_freq'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.estimateTempo(getattr(self, source))

  def estimateTempo(self, source):
    timedomain = sp.array(
        [source[:, i].sum() for i in range(sp.shape(source)[1])])
    self.timedomain = timedomain
    self.timedomain_time = sp.arange(
        0, self.duration, float(self.duration)/timedomain.size)

    tempo_scale, tempo_list = self.generateTempoScale()
    
    #correlating
    corr_list = sp.array(
        [np.correlate(tempo_scale[i], timedomain, 'same').max()
        for i in range(sp.shape(tempo_scale)[0])])
    tempo_max = tempo_list[sp.argmax(corr_list)]

    print 'tempo:', tempo_max, 'BPM'
    self.corr_list = corr_list
    self.tempo_list = tempo_list
    self.wavetempo = tempo_max

  def generateTempoScale(self, sttempo=60, endtempo=300, tempo_step=1):
    fs = self.timedomain.size/self.duration
    t = sp.arange(0, 60/sttempo, 1/fs)
    tempo_list = sp.arange(sttempo, endtempo+tempo_step, tempo_step)
    freq_list = tempo_list / 60.

    tempo_scale = sp.array(
        [sp.absolute(sp.cos(2*sp.pi*freq*t)) for freq in freq_list])
    
    return tempo_scale, tempo_list
    

 
