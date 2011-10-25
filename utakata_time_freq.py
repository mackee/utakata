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


class GenerateMMLTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - MMLデータを生成"""
  def __init__(
      self, prevHandler, target='time_freq',
      window=None, cut_num=40, output_form='MML'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.output_form = output_form
    self.genMML(getattr(self, target), window, cut_num)

  def genMML(self, target, window, cut_num):
    self.interval_list_left = self.extractPhoneme(target, 0, cut_num, window)
    self.interval_list_right = self.extractPhoneme(target, cut_num, 88, window)


    fs = sp.shape(target)[1] / self.duration
    self.mml_left = self.convertMMLList(self.interval_list_left, fs)
    self.mml_right = self.convertMMLList(self.interval_list_right, fs)

    self.mml_text_left = self.convertMMLText(self.mml_left)
    self.mml_text_right = self.convertMMLText(self.mml_right)
    print self.mml_text_left
    print self.mml_text_right
    
  
  def extractPhoneme(self, target, stkey, endkey, window):
    target_other = sp.copy(target[stkey:endkey, :])
    fs = sp.shape(target)[1] / self.duration
    
    if(window == None):
      window = 1
    else:
      window = int(self.wavetempo / 60 * window * fs)
    
    before = target[:, 0]
    interval_list = []
    interval = interval_copy = []
    before_interval = []
    plength = 1
    i = 1
    while(i < sp.shape(target_other)[1]):
      #for c in before_interval:
      #  target_other[c-stkey, i] = 0
      #  before[c-stkey] = 0
      if(sp.array_equal(target_other[:, i], before)):
        plength += 1
        i += 1
      else:
        interval_list.append((interval, plength))
        before_interval = interval_copy
        plength = 1
        freq_data = sp.zeros_like(target_other[:, i])
        if(i+window > sp.shape(target_other)[1]):
          window = sp.shape(target_other)[1] - i
        for j in range(i, i+window):
          freq_data = sp.logical_or(freq_data, target_other[:, j])
        interval = self.analysisInterval(freq_data, stkey)
        interval_copy = interval[:]
        #delete overlap element
        for element in before_interval:
          if element in interval_copy:
            interval.remove(element)
        before = target_other[:, i]
        i += window
    return interval_list


  def analysisInterval(self, freq_data, add_num):
    interval = []
    for i in range(sp.size(freq_data)):
      if(freq_data[i] == 1):
        interval.append(i+add_num)
    return interval
  
  def convertMMLList(self, interval_list, fs):
    note_name, note_length = self.generateNoteLength(self.wavetempo, fs)
    note_scales = self.generateNoteScales()
    mml_list = []
    for interval in interval_list:
      for name, length in zip(note_name, note_length):
        if(interval[1] <= length[1] and interval[1] > length[0]):
          scales = []
          for scale in interval[0]:
            scales.append(note_scales[scale])
          if scales == []:
            scales = ['r']
          mml_list.append((scales, name))
          break
    return mml_list
      


  def generateNoteLength(self, tempo, fs):
    length = (60. / tempo) * fs
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
      note_name = ['1', '1', '8', '8', '4', '4', '2', '2d', '0']
    return (note_name, note_length_pair)

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
 
  def convertMMLText(self, mml_list):
    text = ''
    if(self.output_form == 'MML'):
      octave_character = ('<', '>')
    elif(self.output_form == 'PMX'):
      octave_character = ('+', '-')
    now_octave = 3
    for elements in mml_list:
      phoneme = ''
      if(elements[0] == ['r']):
        phoneme = 'r' + elements[1]
      else:
        length = elements[1]
        for element in elements[0]:
          if(self.output_form == 'MML'):
            if(element[1] < now_octave):
              phoneme += octave_character[0] * (now_octave-element[1])
              now_octave = element[1]
            elif(element[1] > now_octave):
              phoneme += octave_character[1] * (element[1]-now_octave)
              now_octave = element[1]

          if(self.output_form == 'MML'):
            phoneme += element[0] + '0'
          elif(self.output_form == 'PMX'):
            phoneme += element[0] + length + str(element[1]+1) + ' z'
            length = ''

        if(self.output_form == 'MML'):
          phoneme = phoneme[:-1] + elements[1]
        elif(self.output_form == 'PMX'):
          phoneme = phoneme[:-2]

      text += phoneme
      if(self.output_form == 'PMX'):
        text += ' '
    return text


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
    

 
