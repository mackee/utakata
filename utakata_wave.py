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


class CutTopAndBottomSilenceWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 先頭と終端の無音部分をカット"""
  def __init__(self, prevHandler):
    """constructor at CutTopAndBottomSilenceWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.cut()

  def cut(self):
    average = sp.sum(sp.absolute(self.data))/sp.size(self.data)
    head = sp.nonzero(sp.absolute(self.data)>average)[0][5]
    bottom = sp.nonzero(sp.absolute(self.data)>average)[0][-1]
    self.data = self.data[head:bottom]
    self.duration_list = self.duration_list[head:bottom]
    self.duration = self.duration_list[-1] - self.duration_list[0]


class NormalizationWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 正規化"""
  def __init__(self, prevHandler, source='data', output='normalized'):
    """constructor at NormalizationWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.normalization(getattr(self, source), output)

  def normalization(self, source, output):
    self.normalize_factor  = sp.absolute(source).max()
    setattr(self, output,  self.data / sp.absolute(source).max())
    

class DenormalizationWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 正規化"""
  def __init__(self, prevHandler, source='data', output='denormalized'):
    """constructor at NormalizationWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.denormalization(getattr(self, source), output)

  def denormalization(self, source, output):
    setattr(self, output, source * self.normalize_factor)
    print getattr(self, output)



class WithoutToBiasWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - バイアスをなしにする"""
  def __init__(self, prevHandler):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.withoutToBias()

  def withoutToBias(self):
    self.data = self.data - sp.sum(self.data[0:100])/100


#@stopwatch
class GaborwaveletWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - ガボールウェーブレット変換"""
  def __init__(self, prevHandler, target='data'):
    """constructor at GaborwaveletWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    target = getattr(self, target)
    self.generateGaborMotherWavelet()
    self.convertToTimeAndFrequencyData(85, target)

  def generateGaborMotherWavelet(self):
    pitch = 440.0
    sigma = 6.
    NL = 48
    NU = 39
    print 'sampling rate:', self.fs, 'Hz'
    fs = float(self.fs)
    self.sample_duration = 10.
    #asigma = 0.3
    limit_t = 0.1
    #zurashi = 1.

    #NS = NL + NU + 1
    f = sp.array([2**(i/12.) for i in range(NL+NU+1)]) * pitch*2**(-NL/12.)
    f = f[:, sp.newaxis]
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

  def convertToTimeAndFrequencyData(self, grain, target):
    d = self.sample_duration
    length = max(sp.shape(sp.arange(1, sp.size(target) - sp.size(self.gabor[1]), d)))
    scale = sp.zeros((88, length))
    datacapsule = sp.zeros((sp.shape(self.gabor)[1], grain))

    # 行列を束ねて処理
    #   個々にgabor*datadataを計算すると時間がかかる
    #   一気にdatadataを束ねるとメモリ消費量が半端ない
    #   よってdatadataに定義された数だけ束ねて処理
    m = 0
    datasize = sp.size(target) - sp.size(self.gabor[1])

    for k in sp.arange(1, datasize+1, d*grain):
      capsule_pointer = 0
      endl = k+d*grain

      if endl > datasize:
        endl = k + datasize%(d*grain)

      for l in sp.arange(k, endl, d):
        datadata = target[l:l+sp.size(self.gabor[1])]
        datacapsule[:, capsule_pointer] = datadata
        capsule_pointer += 1

      try:
        scale[:, m:m+grain] = sp.absolute(
            sp.dot(self.gabor,datacapsule[:, :capsule_pointer]))
      except ValueError:
        pass
      m += grain
    
    self.time_freq = scale

#@stopwatch
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


#@stopwatch
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


class AbsoluteWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 絶対値を取る"""
  def __init__(self, prevHandler, source='data'):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.absolute(getattr(self, source))

  def absolute(self, source):
    self.absoluted = sp.absolute(source)

class DiffWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 前後との差分を取る"""
  def __init__(self, prevHandler, source='data', n=1):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.diff(getattr(self, source), n)

  def diff(self, source, n):
    self.diffed = np.diff(self.data, n)

class BinarizeWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 2値化する"""
  def __init__(self, prevHandler, source='data', threshold=None):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.binarize(getattr(self, source),threshold)

  def binarize(self, source, threshold):
    if(threshold == None):
      threshold = sp.sum(source) / sp.shape(source)[0]
    print('threshold:', threshold)
    self.binarized = sp.where(source >= threshold, 1, 0)

class RisingEdgeWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 2値化する"""
  def __init__(self, prevHandler, source='data'):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.risingEdge(getattr(self, source))

  def risingEdge(self, source):
    self.diffed = sp.diff(source)
    self.rising_edged = sp.where(self.diffed > 0, 1, 0)

class CuttingPhenomeWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 空白域を参考に音を切り分ける"""
  def __init__(
      self, prevHandler, source='data', real_data='data', 
      duration_list='duration_list', whitespace=1):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.cutting(
        getattr(self, source), getattr(self, real_data),
        getattr(self, duration_list), whitespace)

  def cutting(self, source, real_data, duration_list, whitespace):
    count = 0
    cutted = []
    cutted_hold = []
    prev_i = 0
    for i in range(sp.shape(source)[0]):
      if(source[i] == 0):
        count += 1
      else:
        if(count > whitespace):
          cutted.append(real_data[prev_i:i])
          cutted_hold.append(duration_list[prev_i:i])
          prev_i = i
        count = 0
    cutted.append(real_data[prev_i:i])
    cutted_hold.append(duration_list[prev_i:i])

    self.cutted = cutted
    self.cutted_hold = cutted_hold
    print len(self.cutted)

class SavePlotWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 画像としてグラフを保存する"""
  def __init__(
      self, prevHandler, source='data', hold=None):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    if(hold == None):
      self.savePlot(getattr(self, source), None)
    else:
      self.savePlot(getattr(self, source), getattr(self, hold))

  def savePlot(self, source, hold):

    for i, data in enumerate(source):
      try:
        self.figure += 1
      except AttributeError:
        self.figure = 1
      plt.figure(self.figure)
      
      if(hold == None):
        plt.plot(data)
      else:
        if(isinstance(hold, list)):
          plt.axis([hold[i][0], hold[i][-1], -1.0, 1.0])
          print(i, data, hold[i][0], hold[i][-1])
          #plt.ylim(-1.0, 1.0)
          plt.plot(hold[i], data)
        else:
          plt.axis([hold[0], hold[-1], -1.0, 1.0])
          plt.plot(hold, data)
      
      plt.savefig('image/' + str(i) + '.png', dpi=72)
      plt.delaxes()


class SaveFileWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - Waveファイルとして保存する"""
  def __init__(
      self, prevHandler, source, filename):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.saveWavefile(getattr(self, source), filename)
  
  def saveWavefile(self, source, filename):
    data = source.astype(sp.int16)

    wavefile = wave.open(filename, 'w')
    params = (1, 2, 44100, len(data)/2, 'NONE', 'not compressed')
    wavefile.setparams(params)
    wavefile.writeframes(data.tostring())

    wavefile.close()

class SaveFileMultiWavedataHandler(SaveFileWavedataHandler):
  """Wave数列に対するハンドラ - 配列を複数のWaveファイルとして保存する"""
  def __init__(
      self, prevHandler, sources, fileprefix):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.saveWavefileMulti(getattr(self, sources), fileprefix)
  
  def saveWavefileMulti(self, sources, fileprefix):
    for i, source in enumerate(sources):
      self.saveWavefile(source*self.normalize_factor, fileprefix+str(i)+'.wav')
 

class ConvertDTypeWavedataHandler(BaseProcessHandler):
  """Wave数列に対するハンドラ - 数列の型変換"""
  def __init__(
      self, prevHandler, target, output, dtype):
    """constructor at WithoutToBiasWavedataHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    self.convertDType(getattr(self, target), output, dtype)

  def convertDType(self, target, output, dtype):
    setattr(self, output, target.astype(getattr(sp, dtype)))
    print 'convert to ' + type(getattr(self, output))
