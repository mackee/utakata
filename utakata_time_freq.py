# -*- coding:utf8 -*-
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import scipy.linalg as slng

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
  def __init__(self, prevHandler, source_name='time_freq', extent='default'):
    BaseProcessHandler.__init__(self, prevHandler)
    source = getattr(self, source_name)
    self.plot(source, extent)

  def plot(self, source, extent):
    try:
      self.figure += 1
    except AttributeError:
      self.figure = 1
    
    plt.figure(self.figure)
    if(extent == 'default'):
      extent = [0, self.duration, 0, sp.shape(source)[0]]
      plt.xlabel('time[sec]')
    elif(extent == None):
      extent = None
      plt.xlabel('frame')
    plt.ylabel('key')
    plt.imshow(source, aspect='auto', origin='lower', extent=extent)


class SavePlotTimeFreqDataHandler(PlotTimeFreqDataHandler):
  """時間周波数データに対するハンドラ - グラフを画像に保存する"""
  def __init__(self, prevHandler, savefile, source_name='time_freq', extent='default'):
    PlotTimeFreqDataHandler.__init__(self, prevHandler, source_name, extent)
    plt.savefig(savefile, dpi=72)
    plt.delaxes()


class Plot3DTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 3Dグラフにプロットする"""
  def __init__(self, prevHandler, source='time_freq', extent='default'):
    BaseProcessHandler.__init__(self, prevHandler)
    source = getattr(self, source)
    self.plot3D(source, extent)

  def plot3D(self, source, extent):
    try:
      self.figure += 1
    except AttributeError:
      self.figure = 1

    x = sp.arange(0, sp.shape(source)[1])
    y = sp.arange(0, sp.shape(source)[0])
    X, Y = sp.meshgrid(x, y)
    Z = source

    fig = plt.figure(self.figure)
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, cmap=cm.jet)


class MultipleTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 行列と行列との積をとる"""
  def __init__(self, prevHandler,
      target, load_matrix, output='multiplied', pinv=False, maximum=None):
    BaseProcessHandler.__init__(self, prevHandler)
    self.load_matrix = getattr(self, load_matrix)
    if(pinv):
      self.load_matrix = slng.pinv(self.load_matrix)
    self.multiple(getattr(self, target), output, maximum)

  def multiple(self, target, output, maximum=None):
    multiplied = sp.dot(self.load_matrix, target)
    if(maximum != None):
      multiplied = sp.maximum(multiplied, maximum)
    setattr(self, output, multiplied)


class MultipleMatfileAndTimeFreqDataHandler(MultipleTimeFreqDataHandler):
  """時間周波数データに対するハンドラ - MATLABファイルの行列との積をとる"""
  def __init__(self, prevHandler,
      target, matrix_file, output='multiplied', pinv=False, maximum=None):
    BaseProcessHandler.__init__(self, prevHandler)
    self.importMatrixFile(matrix_file)
    if(pinv):
      self.load_matrix = slng.pinv(self.load_matrix)
    self.multiple(getattr(self, target), output, maximum)

  def importMatrixFile(self, matrix_file):
    self.load_matrix = sio.loadmat(matrix_file)[matrix_file]


class BinarizeTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 二値化をする"""
  def __init__(self, prevHandler, threshold, target_data='time_freq'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.binarize(getattr(self, target_data), threshold)

  def binarize(self, target_data, threshold):
    data = sp.maximum(target_data, threshold)
    sp.putmask(data, data>threshold, 1.)
    sp.putmask(data, data<=threshold, 0.)
    self.binarized_data = data


class NormalizeTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 正規化をする"""
  def __init__(
      self, prevHandler, target='time_freq', split=39,
      set_name='time_freq', window=None):
    BaseProcessHandler.__init__(self, prevHandler)
    setattr(
        self, set_name,
        self.normalizeLR(getattr(self, target), window, split))

  def normalizeLR(self, target, window, split):
    leftroll = self.normalize(target[:split, :], window)
    rightroll = self.normalize(target[split:, :], window)
    return sp.r_[leftroll, rightroll]

  def normalize(self, target, window):
    if(window == None):
      return target / target.max()
    
    else:
      if(window <= 1):
        fs = sp.shape(target)[1] / self.duration
        window_data = fs * window
      else:
        window_data = window
      
      normalized = sp.copy(target)
      for i in range(0, sp.shape(normalized)[1], int(window_data)):
        max_value = normalized[:, i:i+window_data].max()
        normalized[:, i:i+window_data] /= max_value

      return normalized
 

class LogTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 対数を取る"""
  def __init__(self, prevHandler):
    BaseProcessHandler.__init__(self, prevHandler)
    self.log()

  def log(self):
    self.time_freq = sp.log10(self.time_freq)

class CutOffNoiseTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - テンポ情報をもとにノイズ除去"""
  def __init__(self, prevHandler, source='binarized', output='antinoised', minnotel=1./4.):
    BaseProcessHandler.__init__(self, prevHandler)
    setattr(self, output, self.scanSound(getattr(self, source), minnotel))

  def scanSound(self, source, minnotel):
    binarized = source
    scale = 60. / self.wavetempo * (binarized[0].size / self.duration)
    noise_length = scale*minnotel

    antinoised = sp.zeros_like(binarized)

    for i in range(sp.shape(binarized)[0]):
      new_line = binarized[i, :].copy()
      diffed = sp.diff(new_line)
      ones_keys = sp.where(diffed == 1)[0]
      minus_keys = sp.where(diffed == -1)[0]
      
      if(ones_keys.size != 0 and minus_keys.size != 0):
        if(ones_keys[0] > minus_keys[0]):
          new_line = self.cutNoise(
              (0, minus_keys[0]), noise_length, new_line)
          minus_keys = sp.delete(minus_keys, 0)

        if(ones_keys[-1] > minus_keys[-1]):
          new_line = self.cutNoise(
              (ones_keys[-1], new_line.size-1), noise_length, new_line)
          ones_keys = sp.delete(ones_keys, -1)

        for j in range(sp.size(ones_keys)):
          new_line = self.cutNoise(
              (ones_keys[j], minus_keys[j]), noise_length, new_line)

        antinoised[i, :] = new_line

    return antinoised


  def cutNoise(self, keys, noise_length, line):
    if(keys[1] - keys[0] < noise_length):
      array_ranges = sp.arange(keys[0], keys[1]+1)
      line[array_ranges] = sp.zeros_like(array_ranges)
    else:
      pass
    return line


class GradOnPitchTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 音程が高くなるにつれて強調する"""
  def __init__(self, prevHandler, factor, coef):
    BaseProcessHandler.__init__(self, prevHandler)
    self.gradTimeFreq(factor, coef)

  def gradTimeFreq(self, factor, coef):
    tf = self.time_freq
    time_freq = [tf[i, :]*(i**(1./factor))*(coef/(i+1)) for i in range(sp.shape(tf)[0])]
    self.time_freq = sp.array(time_freq)


class EmphasizePitchTimeFreqDataHandler(BaseProcessHandler):
  def __init__(self, prevHandler, target, emrange, factor, output):
    BaseProcessHandler.__init__(self, prevHandler)
    target = getattr(self, target)
    if len(emrange) == 2:
      emrange = (emrange[0], emrange[1], 0, target.shape[1])
    emphasized = self.emphasizePitch(target, emrange, factor)
    setattr(self, output, emphasized)

  def emphasizePitch(self, target, emrange, factor):
    target[emrange[0]:emrange[1], emrange[2]:emrange[3]] = target[
        emrange[0]:emrange[1], emrange[2]:emrange[3]
      ] * factor
    return target


class GenerateScoreTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - MMLデータを生成"""
  def __init__(
      self, prevHandler, target='time_freq',
      cut_num=40, output_form='MML'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.output_form = output_form

    self.left_score = self.generateScore(
        getattr(self, target)[:cut_num, :], cut_num)
    self.key_offset = cut_num
    self.prev_key = []
    self.right_score = self.generateScore(
        getattr(self, target)[cut_num:, :], cut_num)
    try:
      if(output_form == 'MML'):
        print self.convertMML(self.left_score)
        print self.convertMML(self.right_score)
      elif(output_form == 'PMX'):
        print self.convertPMX(self.left_score)
        print self.convertPMX(self.right_score)
    except TypeError:
      pass

  def generateScore(self, target, cut_num):
    fs = self.time_freq_fs = sp.shape(target)[1]  / self.duration
    noise_length = 60. / self.wavetempo * fs
    working = sp.copy(target)
    score = []
    while(sp.shape(working)[1] > noise_length):
      cutted, working = self.cutOutToChangingPoint(working)
      note = self.extractNote(cutted, cut_num)
      score.append(note)
    return score

  def cutOutToChangingPoint(self, working):
    timedomain = sp.array(
        [sp.sum(working[:, i]) for i in range(sp.shape(working)[1])])
    init_data = timedomain[0]
    try:
      changing_point = sp.where(timedomain != init_data)[0][0]
    except IndexError:
      changing_point = -1
    cutted = working[:, :changing_point]
    working = working[:, changing_point:]
    
    try:
      for i in self.prev_key:
        cutted[i, :] = sp.zeros_like(cutted[i, :])
    except AttributeError:
      pass

    return cutted, working

  def extractNote(self, cutted, cut_num):
    phonetic = self.calcPhonetic(sp.shape(cutted)[1])
    chord = self.extractChord(cutted)
    return (chord, phonetic)

  def calcPhonetic(self, length):
    note_sample = self.generateNoteLength()
    for name, length_pair in note_sample:
      if(length_pair[0] <= length and length < length_pair[1]):
        return name

  def extractChord(self, cutted):
    key_list = sp.where(cutted[:, 0] == 1)
    chord = self.convertToScale(key_list[0])
    return chord

  def convertToScale(self, key_list):
    try:
      key_offset = self.key_offset
    except AttributeError:
      key_offset = 0

    self.prev_key = key_list
    note_scales = self.generateNoteScales()
    chord = [note_scales[i+key_offset] for i in key_list]
    return chord
      
  def generateNoteLength(self):
    length = (60. / self.wavetempo) * self.time_freq_fs
    note_length = sp.array([2**i for i in range(5)]) / 4.
    note_length *= length
    note_huten = sp.array(
        [note_length[i-1]+note_length[i] for i in range(1, 5)])
    note_length = sp.r_[note_length, note_huten]
    note_length = sp.sort(note_length)

    note_length_pair = []
    for i in range(note_length.size):
      try:
        upper = (note_length[i+1] - note_length[i])/2
        upper += note_length[i]
      except IndexError:
        upper = note_length[i] * 2
      try:
        lower = note_length_pair[-1][1]
      except IndexError:
        lower = 0
      note_length_pair.append((lower, upper))
        
    if(self.output_form == 'MML'):
      note_name = ['16', '16.', '8', '8.', '4', '4.', '2', '2.', '1']
    elif(self.output_form == 'PMX'):
      note_name = ['1', '1d', '8', '8d', '4', '4d', '2', '2d', '0']
    return (note_name, note_length_pair)

  def generateNoteScales(self):
    if(self.output_form == 'MML'):
      scales = ['a', 'a+', 'b', 'c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+']
    elif(self.output_form == 'PMX'):
      scales = ['a', 'as', 'b', 'c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs']
    scales = (scales * 9)[:88]
    octaves = []
    for i in range(9):
      octaves += ([i]*12)
    octaves = octaves[10:98]
    return zip(scales, octaves)
 
  def convertMML(self, score):
    mml = ''
    now_octave = 3

    for note in score:
      if(note[0] == []):
        mml += 'r' + note[1]
      else:
        for scale in note[0]:
          if(scale[1] < now_octave):
            mml += '<' * (now_octave-scale[1])
            now_octave = scale[1]
          elif(now_octave < scale[1]):
            mml += '>' * (scale[1]-now_octave)
            now_octave = scale[1]
          mml += scale[0] + '0'
        mml = mml[:-1] + note[1]
    return mml

  def convertPMX(self, score):
    pmx = ''

    for note in score:
      if(note[0] == []):
        pmx += 'r' + note[1] + ' '
      else:
        length = note[1]
        for scale in note[0]:
          pmx += scale[0] + length + str(scale[1]) + ' z'
          length = ''
        pmx = pmx[:-2] + ' '
    return pmx


class EstimateTempoTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - テンポを推定"""
  def __init__(
    self, prevHandler, source='time_freq'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.estimateTempo(getattr(self, source))

  def estimateTempo(self, source):
    timedomain = sp.array(
        [source[:, i].sum() for i in range(sp.shape(source)[1])])
    self.timedomain = timedomain
    self.timedomain_time = sp.arange(
        0, self.duration, float(self.duration)/timedomain.size)

    tempo_scale, tempo_list = self.generateTempoScale()
    
    #correlating
    corr_list = sp.array(
        [np.correlate(tempo_scale[i], timedomain, 'same').max()
        for i in range(sp.shape(tempo_scale)[0])])
    tempo_max = tempo_list[sp.argmax(corr_list)]

    print 'tempo:', tempo_max, 'BPM'
    self.corr_list = corr_list
    self.tempo_list = tempo_list
    self.wavetempo = tempo_max

  def generateTempoScale(self, sttempo=60, endtempo=300, tempo_step=1):
    fs = self.timedomain.size/self.duration
    t = sp.arange(0, 60/sttempo, 1/fs)
    tempo_list = sp.arange(sttempo, endtempo+tempo_step, tempo_step)
    freq_list = tempo_list / 60.

    tempo_scale = sp.array(
        [sp.absolute(sp.cos(2*sp.pi*freq*t)) for freq in freq_list])
    
    return tempo_scale, tempo_list
 

class AverageTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 平均をとる"""
  def __init__(
    self, prevHandler, source='time_freq', output='averaged'):
    BaseProcessHandler.__init__(self, prevHandler)
    self.averageTimeFreq(getattr(self, source), output)

  def averageTimeFreq(self, source, output):
    line_average = sp.zeros((88, 1))
    for i in range(sp.shape(source)[0]):
      line_average[i][0] = sp.average(source[:][i])
    setattr(self, output, line_average)
 

class SaveMatfileTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - Matfileに行列を保存する"""
  def __init__(
    self, prevHandler, source, filename):
    BaseProcessHandler.__init__(self, prevHandler)
    self.saveMatfile(getattr(self, source), filename)

  def saveMatfile(self, source, filename):
    sio.savemat(filename+'.mat', {filename: source}, oned_as='row')


class SelectTimeDomainTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 任意の時刻での周波数特性を取り出す"""
  def __init__(
    self, prevHandler, source, key, output):
    BaseProcessHandler.__init__(self, prevHandler)
    if 'best' == key:
      key = self.best[0]
    self.selectTimeDomain(getattr(self, source), output, key)

  def selectTimeDomain(self, source, output, key):
    print sp.shape(source[:, key])
    setattr(self, output, source[:, key])


class SelectMostFlatTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - 掛けたときに最も平坦となるような周波数特性を取り出す"""
  def __init__(
      self, prevHandler, source, output):
    BaseProcessHandler.__init__(self, prevHandler)
    self.selectMostFlatData(getattr(self, source), output)

  def selectMostFlatData(self, source, output):
    best = self.scanMostFlatData(source)
    setattr(self, output, best)

  def scanMostFlatData(self, source):
    best = (0, 1) # (フレーム, diff of factor)
    for i in range(sp.shape(source)[0]):
      factor = self.calcFlatFactor(source, i)
      if 1 < factor:
        diff = factor - 1
      else:
        diff = 1 - factor
      if best[1] > diff:
        best = (i, diff)
        print best

    return best

  def calcFlatFactor(self, source, frame):
    """ 平坦かどうかの指数を算出
      :param source: 時間周波数データ n x 88
      :param frame: 用いる時間上の周波数特性
    """
    invframe = 1 / source[:, frame]
    invframe = invframe[:, sp.newaxis]
    calculated = source * invframe
    factor = sp.average(calculated)

    return factor


class NoteOnOffTimeFreqDataHandler(BaseProcessHandler):
  """NoteON, NoteOFFを検出"""
  def __init__(self, prevHandler, source='binarized', output='notes'):
    """constructor at ImportWaveHandler."""
    BaseProcessHandler.__init__(self, prevHandler)
    notes = self.noteOnOff(getattr(self, source))
    setattr(self, output, notes)

  def noteOnOff(self, source):
    notes = sp.zeros((10000, 5))
    nlistnum = 0
    for n in range(1, sp.shape(source)[0]):
      beforestate = source[n, 0]
      bssample = 1
      for k in range(1, sp.shape(source)[1]):
        if beforestate != source[n, k]:
          notes[nlistnum, 0] = bssample
          notes[nlistnum, 1] = k-1
          notes[nlistnum, 2] = n
          notes[nlistnum, 3] = notes[nlistnum, 1] - notes[nlistnum, 0] + 1
          notes[nlistnum, 4] = beforestate
          beforestate = source[n, k]
          bssample = k
          nlistnum += 1
   
    self.nlistnum = nlistnum - 1
    return notes[:nlistnum+2, :]


# NoteListのため分離予定
class AntinoiseNoteListHandler(BaseProcessHandler):
  """NotesListへの加工 - ノイズ除去"""
  def __init__(self, prevHandler, source='notes', output='antinoised'):
    BaseProcessHandler.__init__(self, prevHandler)
    antinoised = self.antinoise(getattr(self, source))
    setattr(self, output, antinoised)

  def antinoise(self, source):
    notes = source
    minnotel = 1./4.
    self.minnotel = minnotel
    shibu = 60. / self.wavetempo * (self.binarized_data[0].size / self.duration)
    self.shibu = shibu
    noiselen = shibu*minnotel
    # d = 10 sampling間隔
    #shibu = (self.fs/self.sample_duration) / (self.wavetempo/60.)
    # minnotel = 1/4 4分音符の1/4 = 16分音符で採譜
    #noiselen = (shibu*1./4.)
    
    k = 0
    while k <= self.nlistnum:
      if notes[k, 2] != notes[k+1, 2]:
        if notes[k, 3] < noiselen:
          notes = sp.r_[notes[0:np.max(k-1, 0), :], notes[k+1:, :]]
      
      if (notes[k, 0] != 0) & (notes[k, 2] == notes[k+2, 2]):
        if notes[k+1, 3] < noiselen:
          notes[k, 1] = notes[k+2, 1]
          notes[k, 3] = notes[k, 1] - notes[k, 0] + 1
          notes = sp.r_[notes[:k, :], notes[k+3:, :]]
          k -= 1

      self.nlistnum = sp.shape(notes)[0] - 4
      k += 1
    
    return notes[:self.nlistnum, :]


class CreateOnNoteListHandler(BaseProcessHandler):
  """NoteListへの加工 - NoteOn Listの作成"""
  def __init__(self, prevHandler, source='antinoised', output='noteOns'):
    BaseProcessHandler.__init__(self, prevHandler)
    noteOns = self.noteOnList(getattr(self, source))
    setattr(self, output, noteOns)

  def noteOnList(self, source):
    antinoised = source
    noteOnCount = 0
    noteOnList = sp.zeros((sp.sum(antinoised[:, 4], axis=0), 4))
    for k in range(self.nlistnum):
      if antinoised[k, 4] == 1:
        noteOnList[noteOnCount, 0:4] = antinoised[k, 0:4]
        noteOnCount += 1
    self.noteOnCount = noteOnCount - 1

    return noteOnList


class CreatePianorollTimeFreqDataHandler(BaseProcessHandler):
  """時間周波数データに対するハンドラ - ピアノロールの作成"""
  def __init__(
      self, prevHandler, size, note_on='noteOns',
      output='pianoroll'):
    BaseProcessHandler.__init__(self, prevHandler)
    if size == 'fixToResolution':
      size = (88, max(self.fixToResolution[:, 1]))
      note_on = 'fixToResolution'
    else:
      size = sp.shape(getattr(self, size))
    pianoroll = self.createPianoroll(size, getattr(self, note_on))
    setattr(self, output, pianoroll)

  def createPianoroll(self, size, noteOns):
    pianoroll = sp.zeros(size)

    for k in range(self.noteOnCount):
      noteNo = noteOns[k, 2]
      noteON = noteOns[k, 0]
      #noteOFF = noteOns[k, 1]
      noteONTime = noteOns[k, 3]
      pianoroll[noteNo, noteON:noteON+noteONTime] = sp.ones(1, noteONTime)

    return pianoroll


class DeleteSilentNoteListHandler(BaseProcessHandler):
  """NoteListへの加工 - 無音部分を削除する"""
  def __init__(self, prevHandler,
      antinoised='antinoised', note_on='noteOns', output='deleteSilents'):
    BaseProcessHandler.__init__(self, prevHandler)
    noteOns = self.deleteSilent(
        getattr(self, antinoised), getattr(self, note_on))
    #setattr(self, output, deleteSilents)
    setattr(self, note_on, noteOns)

  def deleteSilent(self, antinoised, noteOns):
    """
    noiseCutSumVector = sp.sum(antinoised, axis=0)
    for k in range(len(noiseCutSumVector)):
      if k != 0:
        startSamp = k
        break
    deleteSilents = antinoised[:, startSamp:]
    noteOns[0:self.noteOnCount, 0:1] = noteOns[0:self.noteOnCount, 0:1] - startSamp + 1
    """
    noteOns = noteOns[sp.where(noteOns[:, 3] > 0), :][0]

    return noteOns


class NormalizeLengthNoteListHandler(BaseProcessHandler):
  """NoteListへの加工 - 音符の正規化"""
  def __init__(self, prevHandler, note_on='noteOns', output='fixToResolution', factor=0):
    BaseProcessHandler.__init__(self, prevHandler)
    fixToResolution = self.normalizeLength(getattr(self, note_on), factor)
    setattr(self, output, fixToResolution)

  def normalizeLength(self, noteOns, factor):
    #shibu = 60. / self.wavetempo * (self.binarized_data[0].size / self.duration)
    shibu = (self.fs/10.) / (self.wavetempo/60.)
    fixToResolution = noteOns/shibu*480.
    fixToResolution[:, 2] = noteOns[:, 2]
    # MIDI_Res(分解能) = 480
    MIDI_Res = 480.
    minnotel = 1./4.*MIDI_Res
    #rate(許容誤差)
    rate = 0.5

    #NoteNoが大きいものから順に並び替え
    fixToResolution = self.rowsort(fixToResolution)
    self.oldFixToResolution = sp.copy(fixToResolution)

    #lilypond符号用リスト
    book = [[] for i in range(fixToResolution.shape[0])]

    for n in range(fixToResolution.shape[0]):
      x_cor = fixToResolution[n, 0] + minnotel*rate - 1

      #x_cor = fixToResolution[n, 0] + minnotel - 1
      x_cor = (sp.floor(x_cor/minnotel))*minnotel
      if(x_cor == 0):
        x_cor = 1
      fixToResolution[n, 0] = x_cor
      fixToResolution[n, 3], book[n] = self.normalizeNoteLength(fixToResolution[n, 3] + factor)
      book[n] = self.convertNoteNo(fixToResolution[n, 2]) + book[n]
      fixToResolution[n, 1] = fixToResolution[n, 3] + fixToResolution[n, 0] - 1
    
    self.book = book
    return fixToResolution

  def rowsort(self, a):
    result = [[] for i in range(88)]
    #同NoteNoで分割
    for column in a:
      result[int(column[2])].append([c for c in column])
    result = [sp.array(r) for r in result if r != []]
    #NoteNo内でソート
    result = [r[sp.argsort(r[:, 0])[::-1], :] for r in result]
    
    arr = []
    #元の形式に戻す
    for r in result:
      for phenome in r:
        arr.append(phenome)
    arr = sp.array(arr, dtype=int)[::-1, :]
    return arr

  def normalizeNoteLength(self, length):
    nlen = 0
    ov_len = 0
    rhythm = 0
    if length > 1920:
      ov_len = sp.floor(length/1920.)
      length = sp.mod(length, 1920)

    len_table = [
        [0],      # 無音
        #[20],     # 64分3連
        [30],     # 64分
        #[40],     # 32分3連
        [45],     # 付点64分
        [60],     # 32分
        #[80],     # 16分3連
        [90],     # 付点32分
        [120],    # 16分
        #[160],    # 8分3連
        [180],    # 付点16分
        [240],    # 8分
        #[320],    # 4分3連
        [360],    # 付点8分
        [480],    # 4分
        #[640],    # 2分3連
        [720],    # 付点4分
        [960],    # 2分
        #[1280],   # 全音3連
        [1440],   # 付点2分
        [1920],   # 全音符
    ]

    rhythms = [
        [],      # 無音
        #[],     # 64分3連
        ['64'],     # 64分
        #[],     # 32分3連
        ['64.'],     # 付点64分
        ['32'],     # 32分
        #[],     # 16分3連
        ['32.'],     # 付点32分
        ['16'],    # 16分
        #[],    # 8分3連
        ['16.'],    # 付点16分
        ['8'],    # 8分
        #[],    # 4分3連
        ['8.'],    # 付点8分
        ['4'],    # 4分
        #[],    # 2分3連
        ['4.'],    # 付点4分
        ['2'],    # 2分
        #[],   # 全音3連
        ['2.'],   # 付点2分
        ['1'],   # 全音符
    ]

    if length != 0: # <-  if len = n*1920 (n = 1, 2, …) DO NOT calc
      for k in range(1, len(len_table)):
        if length > len_table[k-1]:
          if length <= len_table[k]:
            nlen = len_table[k][0]
            rhythm = rhythms[k][0]
    nlen = nlen + ov_len*1920

    return nlen, rhythm

  def convertNoteNo(self, noteNo):
    pitches = [
      'a,,,,', 'ais,,,,', 'b,,,,',
      'c,,,', 'cis,,,', 'd,,,', 'dis,,,', 'e,,,', 'f,,,',
      'fis,,,', 'g,,,', 'gis,,,', 'a,,,', 'ais,,,', 'b,,,',
      'c,,', 'cis,,', 'd,,', 'dis,,', 'e,,', 'f,,',
      'fis,,', 'g,,', 'gis,,', 'a,,', 'ais,,', 'b,,',
      'c,', 'cis,', 'd,', 'dis,', 'e,', 'f,', 'fis,', 'g,', 'gis,', 'a,', 'ais,', 'b,',
      'c', 'cis', 'd', 'dis', 'e', 'f', 'fis', 'g', 'gis', 'a', 'ais', 'b',
      'c\'', 'cis\'', 'd\'', 'dis\'', 'e\'', 'f\'',
      'fis\'', 'g\'', 'gis\'', 'a\'', 'ais\'', 'b\'',
      'c\'\'', 'cis\'\'', 'd\'\'', 'dis\'\'', 'e\'\'', 'f\'\'',
      'fis\'\'', 'g\'\'', 'gis\'\'', 'a\'\'', 'ais\'\'', 'b\'\'',
      'c\'\'\'', 'cis\'\'\'', 'd\'\'\'', 'dis\'\'\'', 'e\'\'\'', 'f\'\'\'',
      'fis\'\'\'', 'g\'\'\'', 'gis\'\'\'', 'a\'\'\'', 'ais\'\'\'', 'b\'\'\'', 'c\'\'\'\'',
    ]

    return pitches[noteNo]


class ToLilyPondNoteListHandler(BaseProcessHandler):
  def __init__(self, prevHandler, book='book', note_on='fixToResolution', split=39):
    BaseProcessHandler.__init__(self, prevHandler)
    book = getattr(self, book)
    noteOns = getattr(self, note_on)

    LRCount  = self.splitLR(book, noteOns, split)
    print '\\version "2.14.2"'
    print '{'
    print '<<'
    print '\\new Staff {\\clef "treble"', 
    rlytext = self.liliezd(book[:LRCount-1], noteOns[:LRCount-1])
    print rlytext, '}'

    print '\\new Staff {\\clef "bass"', 
    llytext = self.liliezd(book[LRCount:], noteOns[LRCount:])
    print llytext, '}'
    print '>>'
    print '}'

  def liliezd(self, book, noteOns):
    book = sp.array(book)
    # 時間軸上で整列させたインデックス
    l = sp.argsort(noteOns[:, 0])
    timebook = book[l]

    lytext = ''
    for phenome in timebook:
      #if phenome != '[]':
      lytext += phenome + ' '

    return lytext

  def splitLR(self, book, noteOns, split):
    LRSCount = 0
    for k in range(self.noteOnCount):
      if noteOns[k, 2] < split:
        LRSCount = k
        break

    return LRSCount

