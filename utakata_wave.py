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


class PlotWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 現在のWavedataをグラフにプロットする"""
  def __init__(self, prevHandler):
    """constructor at ImportWaveHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.plot()

  def plot(self):
    try:
      self.figure += 1
    except NameError:
      self.figure = 1
    plt.figure(self.figure)
    plt.plot(self.data)


class CutTopSilenceWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 先頭の無音部分をカット"""
  def __init__(self, prevHandler):
    """constructor at CutTopSilenceWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.cutTop()

  def cutTop(self):
    average = sp.sum(sp.absolute(self.data))/sp.size(self.data)
    head = sp.nonzero(sp.absolute(self.data)>average)[0][5]
    self.data = self.data[head:]


class NormalizationWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 正規化"""
  def __init__(self, prevHandler):
    """constructor at NormalizationWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.normalization()

  def normalization(self):
    self.data = self.data / sp.absolute(self.data).max()

class GaborwaveletWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - ガボールウェーブレット変換"""
  @stopwatch
  def __init__(self, prevHandler):
    """constructor at NormalizationWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.generateGaborMotherWavelet()
    self.convertToTimeAndFrequencyData(85)

  def generateGaborMotherWavelet(self):
    pitch = 440.0
    sigma = 6.
    NL = 48.
    NU = 39.
    fs = 44100.
    #asigma = 0.3
    limit_t = 0.1
    #zurashi = 1.

    #NS = NL + NU + 1
    f_t = sp.arange(-NL, NU+1)[:, sp.newaxis]
    f = pitch * sp.power(2, f_t/11.)
    sigmao = sigma*10**(-3)*sp.sqrt(fs/f)
    t = sp.arange(-limit_t, limit_t+1/fs, 1/fs)

    inv_sigmao = sp.power(sigmao, -1)
    inv_sigmao_t = inv_sigmao * t
    t_inv_sigmao2 = sp.multiply(inv_sigmao_t, inv_sigmao_t)
    omega_t = 2*sp.pi*f*t
    gabor = (1/sp.sqrt(2*sp.pi))
    gabor = sp.multiply(gabor, sp.diag(inv_sigmao))
    exps = -0.5*t_inv_sigmao2+sp.sqrt(-1)*omega_t
    self.gabor = gabor*sp.exp(exps)

  def convertToTimeAndFrequencyData(self, grain):
    d = 10.
    length = max(sp.shape(sp.arange(1, sp.size(self.data) - sp.size(self.gabor[1]), d)))
    scale = sp.zeros((88, length))
    datacapsule = sp.zeros((8821, grain))

    # 行列を束ねて処理
    #   個々にgabor*datadataを計算すると時間がかかる
    #   一気にdatadataを束ねるとメモリ消費量が半端ない
    #   よってdatadataに定義された数だけ束ねて処理
    m = 0
    datasize = sp.size(self.data) - sp.size(self.gabor[1])

    for k in sp.arange(1, datasize+1, d*grain):
      capsule_pointer = 0
      endl = k+d*grain

      if endl > datasize:
        endl = k + datasize%(d*grain)

      for l in sp.arange(k, endl, d):
        datadata = self.data[l:l+sp.size(self.gabor[1])]
        datacapsule[:, capsule_pointer] = datadata
        capsule_pointer += 1

      scale[:, m:m+grain] = sp.absolute(
          sp.dot(self.gabor,datacapsule[:, :capsule_pointer]))
      m += grain
    
    self.time_freq = scale



