# -*- coding:utf8 -*-
import scipy as sp
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
  def __init__(self, prevHandler):
    BaseProcessHandler.__init__(self, prevHandler)
    self.normalize()

  def normalize(self):
    self.time_freq = self.time_freq / self.time_freq.max()


class LogTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 対数を取る"""
  def __init__(self, prevHandler):
    BaseProcessHandler.__init__(self, prevHandler)
    self.log()

  def log(self):
    self.time_freq = sp.log(self.time_freq)

class ScanSoundPointTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 音が鳴っている箇所をスキャンする"""
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
      
      if(sp.size(ones_keys) < sp.size(minus_keys)):
        new_line = self.noiseOff(
            (0, minus_keys[0]), noise_length, new_line)
        minus_keys = sp.delete(minus_keys, 0)

      for j in range(sp.size(ones_keys)):
        new_line = self.shaping(
            (ones_keys[j], minus_keys[j]), noise_length, new_line)

      antinoised[i, :] = new_line

    self.antinoised = antinoised


  def shaping(self, keys, noise_length, line):
    if(keys[1] - keys[0] < noise_length):
      array_ranges = sp.arange(keys[0], keys[1]+1)
      line[array_ranges] = sp.zeros_like(array_ranges)
    else:
      pass
    return line




