# -*- coding:utf8 -*-

import scipy as sp
import wave

# 入力ハンドラ
class ImportWavedataHandler:
  """入力ハンドラ - waveファイルをインポートしてscipy数列に変換する"""
  
  def __init__(self, filename):
    """constructor at ImportWaveHandler.
    
    :param filename: import file name::string
    """
    self.filename = filename
    self.data = self.original = self.importWave()

  def importWave(self):
    """Wave file to ndarray"""
    wf = wave.open(self.filename, 'rb')
    waveframes = wf.readframes(wf.getnframes())
    return sp.fromstring(waveframes, sp.int16)


