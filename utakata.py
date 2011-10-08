# -*- coding:utf8 -*-

#import scipy as sp
import matplotlib.pyplot as plt

class SignalDispatcher(object):
  """信号処理をまとめて行うバッチ実行クラス"""

  def __init__(self, input_handler, process_handler_list, option=[]):
    """Constructor at SignalDispatcher.

    :param input_handler: インプットハンドラの名前と引数のタプル
    :param process_handler_list:  処理ハンドラの名前と引数のタプルが
                                  実行順にリストになったもの
    """
    self.input_handler = input_handler
    self.process_handler_list = process_handler_list
    self.option = option

  def importHandler(self):
    """入力ハンドラを利用して信号処理する対象の数列データを入力"""
    self.instance = self.__process(self.input_handler)
  
  def processHandler(self):
    """信号処理ハンドラを逐次実行"""
    for handler in self.process_handler_list:
      handler[1]['prevHandler'] = self.instance
      self.instance = self.__process(handler)
    #グラフプロット
    plt.show()

  def __process(self, handler):
    """generate and exec instance."""
    instance = self.__import(handler)
    return instance

  def __import(self, handler):
    """generate instance method.

    :param handler: turple (handler_name::string, handler_param::kwargs)
    """
    handler_name = handler[0]
    kwargs = handler[1]
    return globals()[handler_name](**kwargs)



