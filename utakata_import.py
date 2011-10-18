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
    self.duration = wf.getnframes() / self.framerate
    data = sp.fromstring(waveframes, sp.int16)
    if(self.fs == None):
      self.fs = self.framerate
    else:
      data = self.resample(data, data.size*(self.fs/self.framerate))
    self.duration_list = sp.arange(0, self.duration, 1./self.fs)
    data = ssig.detrend(data)
    return data
  
  @stopwatch
  def resample(self, data, sample):
    """downsampling."""
    return ssig.resample(data, sample)


