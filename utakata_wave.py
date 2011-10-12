# -*- coding:utf8 -*-
import scipy as sp
import matplotlib.pyplot as plt
#import scipy.io as sio
#import scipy.linalg as slng

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


