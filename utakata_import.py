# -*- coding:utf8 -*-

import scipy as sp
import scipy.signal as ssig
import wave

import time
def stopwatch(wrapped):
  """計時デコレータ"""
  def _wrapper(*args, **kwargs):
    tic = time.time()
    result = wrapped(*args, **kwargs)
    toc = time.time()
    doc = str(wrapped.__doc__).split("\n")[0]
    print("[%s] %f sec" % (doc, toc - tic))
    return result
  return _wrapper


# 入力ハンドラ
class ImportWavedataHandler:
  """入力ハンドラ - waveファイルをインポートしてscipy数列に変換する"""
  
  def __init__(self, filename, fs=None):
    """constructor at ImportWaveHandler.
    
    :param filename: import file name::string
    """
    self.filename = filename
    self.fs = fs
    self.data = self.original = self.importWave()

  def importWave(self):
    """Wave file to ndarray"""
    wf = wave.open(self.filename, 'rb')
    waveframes = wf.readframes(wf.getnframes())
    self.framerate = wf.getframerate()
    data = sp.fromstring(waveframes, sp.int16)
    self.duration = float(wf.getnframes()) / self.framerate
    if(wf.getnchannels() == 2):
      left = sp.array([data[i] for i in range(0, data.size, 2)])
      right = sp.array([data[i] for i in range(1, data.size, 2)])
      left = sp.int32(left); right = sp.int32(right)
      data = sp.int16(left+right) / 2
    if(self.fs == None):
      self.fs = self.framerate
    else:
      #data = self.resample(data, data.size*(self.fs/self.framerate))
      data = ssig.decimate(data, int(self.framerate/self.fs))
    self.duration_list = sp.arange(0, self.duration, 1./self.fs)
    data = ssig.detrend(data)
    return data
  
  @stopwatch
  def resample(self, data, sample):
    """downsampling."""
    return ssig.resample(data, sample)


