# -*- coding:utf8 -*-
import scipy as sp
import scipy.signal as ssig
#import scipy.fftpack as sfft
import numpy as np
import matplotlib.pyplot as plt
import wave
#import scipy.io.wavfile as swav
#import scipy.linalg as slng

import time
def stopwatch(wrapped):
  """計時デコレータ"""
  def _wrapper(*args, **kwargs):
    tic = time.time()
    result = wrapped(*args, **kwargs)
    toc = time.time()
    doc = str(wrapped.__doc__).split("\n")[0]
    print("[%s] %fsec" % (doc, toc - tic))
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


class NoteOnOffPhenomeHandler(BaseProcessHandler):
  """NoteON, NoteOFFを検出"""
  def __init__(self, prevHandler, source='binarized', output='note_list', hold=None):
    """constructor at ImportWaveHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    note_list = self.noteOnOff(getattr(self, source))
    setattr(self, output, note_list)

  def noteOnOff(self, source):
    pass


