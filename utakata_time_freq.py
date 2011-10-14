# -*- coding:utf8 -*-
import scipy as sp
import matplotlib.pyplot as plt
#import scipy.io as sio
#import scipy.linalg as slng

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



