# -*- coding:utf8 -*-

__all__ = ["utakata", "utakata_import", "utakata_wave"]
from utakata import SignalDispatcher
from utakata_import import *
from utakata_wave import *
from utakata_time_freq import *

import time
def stopwatch(self, wrapped):
  """計時デコレータ"""
  def _wrapper(*args, **kwargs):
    tic = time.time()
    result = wrapped(*args, **kwargs)
    toc = time.time()
    doc = str(wrapped.__doc__).split("\n")[0]
    print("[%s] %f[sec]" % (doc, toc - tic))
    return result
  return _wrapper


