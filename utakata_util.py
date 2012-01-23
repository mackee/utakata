# -*- coding:utf8 -*-
import scipy as sp
#import matplotlib.pyplot as plt
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


class GenNoteLengthUtilHandler(BaseProcessHandler):
  """ユーティリティハンドラ - 音符の長さを定義するシーケンスを生成"""
  def __init__(self, prevHandler, tempo_name='wavetempo'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.generateNoteLength(getattr(self, tempo_name))

  def generateNoteLength(self, tempo):
    length = 60 / tempo * self.fs * 4
    note_length = sp.array([2**i for i in range(5)]) / 4
    note_length *= length
    note_huten = sp.array(
        [note_length[i-1]+note_length[i] for i in range(1, 4)])
    note_length = sp.r_[note_length, note_huten]
    self.note_length = sp.sort(note_length)
    self.note_name = ['16', '16.', '8', '8.', '4', '4.', '2', '2.' '1']


class SelectUtilHandler(BaseProcessHandler):
  """ユーティリティハンドラ - リストの中から要素を抜き出して別の変数に入れる"""
  def __init__(self, prevHandler, targets, key, output):
    BaseProcessHandler.__init__(self, prevHandler)
    self.selectElement(targets, key, output)

  def selectElement(self, targets, key, output):
    setattr(self, output, getattr(self, targets)[key])
    print getattr(self, output)

