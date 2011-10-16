# -*- coding:utf8 -*-

import scipy as sp
import scipy.signal as ssig
import wave

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
    self.duration_list = sp.arange(0, self.duration, 1./self.fs)
    data = sp.fromstring(waveframes, sp.int16)
    if(self.fs == None):
      self.fs = self.framerate
    else:
      data = ssig.resample(data, self.duration_list.size)

    return data


