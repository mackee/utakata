# -*- coding:utf8 -*-
import scipy as sp
import scipy.signal as ssig
#import scipy.fftpack as sfft
import numpy as np
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


class PlotWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 現在のWavedataをグラフにプロットする"""
  def __init__(self, prevHandler, source='data', hold=None):
    """constructor at ImportWaveHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    if(hold != None):
      self.plot(getattr(self, source), getattr(self, hold))
    else:
      self.plot(getattr(self, source), None)

  def plot(self, source, hold):
    try:
      self.figure += 1
    except AttributeError:
      self.figure = 1
    plt.figure(self.figure)
    if(hold == None):
      plt.plot(source)
    else:
      plt.plot(hold, source)


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
    self.duration_list = self.duration_list[head:]


class NormalizationWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 正規化"""
  def __init__(self, prevHandler):
    """constructor at NormalizationWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.normalization()

  def normalization(self):
    self.data = self.data / sp.absolute(self.data).max()


class WithoutToBiasWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - バイアスをなしにする"""
  def __init__(self, prevHandler):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.withoutToBias()

  def withoutToBias(self):
    self.data = self.data - sp.sum(self.data[0:100])/100


@stopwatch
class GaborwaveletWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - ガボールウェーブレット変換"""
  def __init__(self, prevHandler):
    """constructor at GaborwaveletWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.generateGaborMotherWavelet()
    self.convertToTimeAndFrequencyData(85)

  def generateGaborMotherWavelet(self):
    pitch = 440.0
    sigma = 6.
    NL = 48.
    NU = 39.
    fs = float(self.fs)
    self.sample_duration = 10.
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
    d = self.sample_duration
    length = max(sp.shape(sp.arange(1, sp.size(self.data) - sp.size(self.gabor[1]), d)))
    scale = sp.zeros((88, length))
    datacapsule = sp.zeros((sp.shape(self.gabor)[1], grain))

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

@stopwatch
class EstimateTempoWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - テンポを推定する"""
  def __init__( self, prevHandler, target_name='wavecorr',
                sttempo=60, endtempo=300, tempo_step=0.5):
    """constructor at ImportWaveHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.estimateTempo( getattr(self, target_name),
                        sttempo, endtempo, tempo_step)

  def estimateTempo(self, target, sttempo, endtempo, tempo_step):
    """Estimate Tempo."""
    self.tempolist = sp.array([])
    t = sp.arange(0, 60/sttempo, 1./self.fs)
    for tempo in sp.arange(sttempo, endtempo, tempo_step):
      f = tempo / 60.
      scale = sp.cos(2*sp.pi*f*t)
      result = np.correlate(scale, target, 'valid')
      result = sp.average(sp.absolute(result))
      self.tempolist = sp.append(self.tempolist, result)
    self.tempolist = sp.absolute(ssig.detrend(sp.log(self.tempolist)))
    self.tempos = sp.arange(sttempo, endtempo, tempo_step)
    self.wavetempo = self.tempos[sp.argmax(self.tempolist)]
    try:
      print 'tempo:',  self.wavetempo, 'BPM'
    except AttributeError:
      pass


@stopwatch
class CalcCorrWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 自己/相互相関を求める"""
  def __init__( self, prevHandler,
                x_name='data', y_name=None, set_name='wavecorr',
                corr_duration=8810*3, corr_num=15000, corr_offset=0 ):
    """constructor at ImportWaveHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    
    self.corr_duration = corr_duration
    self.corr_num = corr_num
    self.corr_offset = corr_offset

    if(y_name != None):
      self.calcCorr(getattr(self, x_name), getattr(self, y_name), set_name=set_name)
    else:
      self.calcCorr(getattr(self, x_name), None, set_name=set_name)
  
  def calcCorr(self, x, y, set_name):
    """ calculate correlation."""
    x_data = x[self.corr_offset:self.corr_offset+self.corr_duration]
    if(y == None):
      y_data = x[self.corr_offset:self.corr_offset+self.corr_duration+self.corr_num]
      #corr = self.correlate(x, x, self.corr_num,
      #                      self.corr_duration, self.corr_offset)
    else:
      y_data = y[self.corr_offset:self.corr_offset+self.corr_duration+self.corr_num]
      #corr = self.correlate(x, y, self.corr_num,
      #                      self.corr_duration, self.corr_offset)
    corr = np.correlate(x_data, y_data, 'same')
    corr = ssig.detrend(corr)
    setattr(self, set_name, corr)

  def correlate(self, x, y, corr_num, corr_duration, corr_offset):
    offset = corr_offset
    cduration = corr_duration
    x_data = x[offset:offset+cduration]
    
    corr_data = sp.array([])
    for i in range(offset, offset+corr_num):
      y_data = y[i:i+cduration]
      corr_data = sp.append(corr_data, sp.dot(x_data, y_data)/y_data.size)
    
    return corr_data

