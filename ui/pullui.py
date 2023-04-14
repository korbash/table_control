import sys
from sys import path
import asyncio

path += ['..']
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,\
     QSlider, QDial, QProgressBar, QLineEdit, QDialog, QDialogButtonBox, QGridLayout
from PyQt6.QtCore import QSize, Qt, pyqtSignal, QLocale
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from qasync import QEventLoop, asyncSlot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np


from mylib import Puller

app = QApplication(sys.argv)

def getDoubleValidator():
    validator = QDoubleValidator()
    validator.setLocale(QLocale("en_US"))
    return validator

class CustomSlider(QSlider):

    doubleValueChanged = pyqtSignal(float)

    def __init__(self, min=0, max=100, pow=3, value=None, *args, **kargs):
        super().__init__(*args, **kargs)
        self.min = min
        self.max = max
        self.pow = pow

        if value is not None:
            self.setValue(value)
        self.valueChanged.connect(self._emitDoubleValueChanged)
        
    def setValue(self, value):
        v = (value - self.min) / (self.max - self.min) * 100
        super().setValue(int(v))
    
    def value(self):
        return np.round(super().value() / 100 * (self.max - self.min) + self.min, self.pow)
    
    def _emitDoubleValueChanged(self):
        self.doubleValueChanged.emit(self.value())

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=10, height=8, dpi=100, label=None) -> None:
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        if not label is None:
            self.axes.set_title(label)
        self.lines = dict()
    
    def addPlot(self, x, y, label, legend=False):
        [self.lines[label]] = self.axes.plot(x, y, label=label)
        if legend:
            self.axes.legend()
        self.axes.grid()

    def changeLine(self, x, y, label):
        if len(x) > len(y):
            x = x[len(x) - len(y):]
        if not label in self.lines:
            self.addPlot(x, y, label=label, legend=(len(self.lines) > 0))
            return
        self.lines[label].set_xdata(x)
        self.lines[label].set_ydata(y)
        # self.lines[label].set_xlim((np.min(x), np.max(x)))
        # self.lines[label].set_ylim((np.min(y), np.max(y)))
        
        self.axes.relim()
        self.axes.autoscale_view()
                                   
        self.draw_idle()

class PIDWindow(QDialog):
    def __init__(self, Kp, Ki, Kd):
        super().__init__()

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        hostLayout = QVBoxLayout()
        mainLayout = QHBoxLayout()
        labelLayout = QVBoxLayout()
        sliderLayout = QVBoxLayout()
        editLayout = QVBoxLayout()
        hostLayout.addLayout(mainLayout)
        mainLayout.addLayout(labelLayout)
        mainLayout.addLayout(sliderLayout)
        mainLayout.addLayout(editLayout)

        labelLayout.addWidget(QLabel('Kp'))
        labelLayout.addWidget(QLabel('Ki'))
        labelLayout.addWidget(QLabel('Kd'))

        self.pSlider = CustomSlider(min=0, max=.01, orientation=Qt.Orientation.Horizontal, value=self.Kp, pow=7)
        self.iSlider = CustomSlider(min=0, max=1, orientation=Qt.Orientation.Horizontal, value=self.Ki, pow=7)
        self.dSlider = CustomSlider(min=-1, max=1, orientation=Qt.Orientation.Horizontal, value=self.Kd, pow=7)
        sliderLayout.addWidget(self.pSlider)
        sliderLayout.addWidget(self.iSlider)
        sliderLayout.addWidget(self.dSlider)

        self.pEdit = QLineEdit(str(self.Kp))
        self.iEdit = QLineEdit(str(self.Ki))
        self.dEdit = QLineEdit(str(self.Kd))
        self.pEdit.setValidator(getDoubleValidator())
        self.dEdit.setValidator(getDoubleValidator())
        self.iEdit.setValidator(getDoubleValidator())
        editLayout.addWidget(self.pEdit)
        editLayout.addWidget(self.iEdit)
        editLayout.addWidget(self.dEdit)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        hostLayout.addWidget(self.buttonBox)

        self.setLayout(hostLayout)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        

        self.pSlider.doubleValueChanged.connect(self.updateKp)
        self.iSlider.doubleValueChanged.connect(self.updateKi)
        self.dSlider.doubleValueChanged.connect(self.updateKd)
        self.pEdit.editingFinished.connect(self.updateKp)
        self.iEdit.editingFinished.connect(self.updateKi)
        self.dEdit.editingFinished.connect(self.updateKd)


    def updateKp(self, Kp=None):
        if Kp is not None:
            self.pEdit.setText(str(Kp))
        else:
            Kp = float(self.pEdit.text())
            self.pSlider.setValue(int(Kp))
        self.Kp = Kp

    def updateKi(self, Ki=None):
        if Ki is not None:
            self.iEdit.setText(str(Ki))
        else:
            Ki = float(self.iEdit.text())
            self.iSlider.setValue(int(Ki))
        self.Ki = Ki

    def updateKd(self, Kd=None):
        if Kd is not None:
            self.dEdit.setText(str(Kd))
        else:
            Kd = float(self.dEdit.text())
            self.dSlider.setValue(int(Kd))
        self.Kd = Kd

class TensWindow(QDialog):
    def __init__(self, puller):
        super().__init__()

        self.Tt = 10
        self.pl = puller

        mainLayout = QVBoxLayout()
        settingsLayout = QHBoxLayout()

        self.tensPlot = MplCanvas()
        mainLayout.addWidget(self.tensPlot)

        self.tensInput = QLineEdit(str(self.Tt))
        self.tensInput.setValidator(getDoubleValidator())
        self.setButton = QPushButton('Set')
        settingsLayout.addWidget(self.tensInput)
        settingsLayout.addWidget(self.setButton)
        mainLayout.addLayout(settingsLayout)

        self.setLayout(mainLayout)

        self.setButton.pressed.connect(self.pull)

    @asyncSlot()
    async def pull(self):
        self.Tt = float(self.tensInput.text())
        self.setButton.setEnabled(False)
        x = []
        y = []
        async for i in self.pl.SetW(self.Tt, dw=0.1):
            x.append(len(x))
            y.append(i)
            self.tensPlot.changeLine(x, y, 'tension')
            await asyncio.sleep(.01)
        self.setButton.setEnabled(True)

class PullWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # pulling parameters
        self.v = 5
        self.a = 5
        self.T0 = 10
        self.w = 100
        self.burnerH = 0
        self.dhKof = 1
        self.iterations = 50

        self.Kp = .0011
        self.Ki = .05
        self.Kd = -.02

        self.ended=False
        self.burnPosFl = False

        # layouts
        mainLayout = QVBoxLayout()

        progressLayout = QHBoxLayout()
        plotLayout = QVBoxLayout()
        settingsLayout = QHBoxLayout()
        sliderLayout = QHBoxLayout()
        subsettingsLayout = QVBoxLayout()

        mainLayout.addLayout(progressLayout)
        mainLayout.addLayout(plotLayout)
        mainLayout.addLayout(settingsLayout)
        settingsLayout.addLayout(sliderLayout)
        settingsLayout.addLayout(subsettingsLayout)

        # widgets
        self.zeroButton = QPushButton('Zero')
        self.mtsButton = QPushButton('MTS')
        self.tensButton = QPushButton('Pull')
        self.progressBar = QProgressBar()
        self.progressBar.setMaximum(self.iterations)
        self.progressText = QLabel('0/')
        self.interationsInput = QLineEdit(str(self.iterations))
        self.interationsInput.setValidator(QIntValidator())
        self.interationsInput.setFixedWidth(50)
        self.stopButton = QPushButton('START')
        progressLayout.addWidget(self.zeroButton)
        progressLayout.addWidget(self.mtsButton)
        progressLayout.addWidget(self.tensButton)
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.progressText)
        progressLayout.addWidget(self.interationsInput)
        progressLayout.addWidget(self.stopButton)

        self.powerPlot = MplCanvas(label='Power')
        self.tensionPlot = MplCanvas(label='Tension')
        self.motionPlot = MplCanvas(label='Motors')
        plotLayout.addWidget(self.powerPlot)
        plotLayout.addWidget(self.tensionPlot)
        plotLayout.addWidget(self.motionPlot)

        self.vSlider = CustomSlider(min=0.1, max=15, orientation=Qt.Orientation.Horizontal, value=self.v)
        self.vInput = QLineEdit(str(self.v))                                                   
        self.vInput.setFixedWidth(100)                                                         
        self.vInput.setValidator(getDoubleValidator())
        self.aSlider = CustomSlider(min=.1, max=15, orientation=Qt.Orientation.Horizontal, value=self.a)
        self.aInput = QLineEdit(str(self.a))                                                   
        self.aInput.setFixedWidth(100)                                                         
        self.aInput.setValidator(getDoubleValidator())
        self.TSlider = CustomSlider(min=0, max=200, orientation=Qt.Orientation.Horizontal, value=self.T0)
        self.TInput = QLineEdit(str(self.T0))
        self.TInput.setFixedWidth(100)
        self.TInput.setValidator(getDoubleValidator())
        self.bhSlider = CustomSlider(min=-10, max=2.5, orientation=Qt.Orientation.Horizontal, value=self.burnerH)
        self.bhInput = QLineEdit(str(self.burnerH))
        self.bhInput.setFixedWidth(100)
        self.bhInput.setValidator(getDoubleValidator())
        self.dhSlider = CustomSlider(min=0, max=3, orientation=Qt.Orientation.Horizontal, value=self.dhKof)
        self.dhInput = QLineEdit(str(self.dhKof))
        self.dhInput.setFixedWidth(100)
        self.dhInput.setValidator(getDoubleValidator())

        sliderSubLayout = QGridLayout()
        sliderSubLayout.addWidget(QLabel('v'), 0, 0)
        sliderSubLayout.addWidget(self.vSlider, 0, 1)
        sliderSubLayout.addWidget(self.vInput, 0, 2)
        sliderSubLayout.addWidget(QLabel('a'), 1, 0)
        sliderSubLayout.addWidget(self.aSlider, 1, 1)
        sliderSubLayout.addWidget(self.aInput, 1, 2)
        sliderSubLayout.addWidget(QLabel('T0'), 2, 0)
        sliderSubLayout.addWidget(self.TSlider, 2, 1)
        sliderSubLayout.addWidget(self.TInput, 2, 2)
        sliderSubLayout.addWidget(QLabel('bH'), 3, 0)
        sliderSubLayout.addWidget(self.bhSlider, 3, 1)
        sliderSubLayout.addWidget(self.bhInput, 3, 2)
        sliderSubLayout.addWidget(QLabel('dh'), 4, 0)
        sliderSubLayout.addWidget(self.dhSlider, 4, 1)
        sliderSubLayout.addWidget(self.dhInput, 4, 2)
        sliderLayout.addLayout(sliderSubLayout)

        self.wDial = QDial(minimum=50, maximum=10000, value=self.w, singleStep=.5)
        self.wDial.setFixedHeight(100)
        self.wInput = QLineEdit(str(self.w))
        self.wInput.setValidator(QIntValidator())
        self.PIDButton = QPushButton('PID settings')
        subsettingsLayout.addWidget(QLabel('window'))
        subsettingsLayout.addWidget(self.wDial)
        subsettingsLayout.addWidget(self.wInput)
        subsettingsLayout.addWidget(self.PIDButton)

        self.burnerButton = QPushButton('Move Burner (up)')
        subsettingsLayout.addWidget(self.burnerButton)
        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

        self.interationsInput.textChanged.connect(self.onNewIterations)
        self.vSlider.doubleValueChanged.connect(self.onNewVSlider)
        self.aSlider.doubleValueChanged.connect(self.onNewASlider)
        self.TSlider.doubleValueChanged.connect(self.onNewTSlider)
        self.bhSlider.doubleValueChanged.connect(self.onNewbh)
        self.dhSlider.doubleValueChanged.connect(self.onNewdh)
        self.wDial.valueChanged.connect(self.onNewWDial)
        self.vInput.returnPressed.connect(self.onNewVSlider)
        self.aInput.returnPressed.connect(self.onNewASlider)
        self.TInput.returnPressed.connect(self.onNewTSlider)
        self.wInput.returnPressed.connect(self.onNewWDial)
        self.bhInput.returnPressed.connect(self.onNewbh)
        self.dhInput.returnPressed.connect(self.onNewdh)

        self.PIDButton.pressed.connect(self.callPIDSettings)
        self.burnerButton.pressed.connect(self.changeBurnerPos)

    def setProgress(self, progress):
        self.progressBar.setValue(progress)
        self.progressText.setText(f'{progress}/')

    def updateAllPlots(self, data):
        if not 'time' in data:
            return
        if 'power' in data:
            self.powerPlot.changeLine(data['time'], data['power'], 'power')
        if 'tension' in data:
            self.tensionPlot.changeLine(data['time'], data['tension'], 'tension')
        if 'tensionWgl' in data:
            self.tensionPlot.changeLine(data['time'], data['tensionWgl'], 'tensionWgl')
        if 'tensionEXPgl' in data:
            self.tensionPlot.changeLine(data['time'], data['tensionEXPgl'], 'tensionEXPgl')
        if 'motorL' in data:
            self.motionPlot.changeLine(data['time'], data['motorL'], 'motorL') 
        if 'motorR' in data:
            self.motionPlot.changeLine(data['time'], data['motorR'], 'motorR')  
        if 'motorM' in data:
            self.motionPlot.changeLine(data['time'], data['motorM'], 'motorM')  
        
    def callPIDSettings(self):
        PIDs = PIDWindow(self.Kp, self.Ki, self.Kd)
        if PIDs.exec():
            self.Kp = PIDs.Kp
            self.Ki = PIDs.Ki
            self.Kd = PIDs.Kd
    
    def onNewVSlider(self, v=None):
        if v is not None:
            self.vInput.setText(str(v))
        else:
            v = float(self.vInput.text())
            self.vSlider.setValue(v)
        self.v = v
    def onNewASlider(self, a=None):
        if a is not None:
            self.aInput.setText(str(a))
        else:
            a = float(self.aInput.text())
            self.aSlider.setValue(a)
        self.a = a       
    def onNewTSlider(self, T=None):
        if T is not None:
            self.TInput.setText(str(T))
        else:
            T = float(self.TInput.text())
            self.TSlider.setValue(T)
        self.T0 = T       
    def onNewWDial(self, w=None):
        if w is not None:
            self.wInput.setText(str(w))
        else:
            w = float(self.wInput.text())
            self.wDial.setValue(w)
        self.w = w
    def onNewbh(self, burnerH=None):
        if burnerH is not None:
            self.bhInput.setText(str(burnerH))
        else:
            burnerH = float(self.bhInput.text())
            self.bhSlider.setValue(burnerH)
        self.burnerH = burnerH
    def onNewdh(self, dhKof=None):
        if dhKof is not None:
            self.dhInput.setText(str(dhKof))
        else:
            dhKof = float(self.dhInput.text())
            self.dhSlider.setValue(dhKof)
        self.dhKof = dhKof
    def onNewIterations(self, num):
        if len(num) == 0:
            return
        self.iterations = int(num)
        self.progressBar.setMaximum(num) 
    def onNewPIDCoefs(self, Kp, Ki, Kd):
        self.Kp = Kp / 100
        self.Ki = Ki
        self.Kd = Kd

    def changeBurnerPos(self):
        self.burnPosFl = not self.burnPosFl
        self.burnerButton.setText('Move burner (down)' if self.burnPosFl else 'Move burner (up)')

    
class ZeroDialog(QDialog):
    def __init__(self, val):
        super().__init__()
        self.setWindowTitle('Last weight shift')


        mLayout = QVBoxLayout()

        message = QLabel(str(val))
        mLayout.addWidget(message)        

        buttonLayout = QHBoxLayout()
        self.acceptButton = QPushButton('OK')
        self.retryButton = QPushButton('Retry')
        buttonLayout.addWidget(self.acceptButton)
        buttonLayout.addWidget(self.retryButton)
        mLayout.addLayout(buttonLayout)

        self.setLayout(mLayout)

        self.acceptButton.pressed.connect(self.accept)
        self.retryButton.pressed.connect(self.reject)

loop = QEventLoop(app)
asyncio.set_event_loop(loop)

window = PullWindow()
window.show()

pl = Puller()
pl.win = window

def zeroTs():
    while True:
        pl.tg.SetZeroWeight(tau=3)
        pl.tg.SetZeroWeight(tau=3)
        res = pl.tg.SetZeroWeight(tau=4) 

        zd = ZeroDialog(res)
        if zd.exec():
            break


window.zeroButton.pressed.connect(zeroTs)

def pullCall():
    pullWin = TensWindow(pl)
    pullWin.exec()

window.tensButton.pressed.connect(pullCall)

@asyncSlot()
async def mts():
    window.mtsButton.setEnabled(False)
    await pl.ms.MoveToStart(moveM=True, zapas=10)
    window.mtsButton.setEnabled(True)

async def plStart():
    async with pl:
        pass

@asyncSlot()
async def start():
    window.stopButton.setText('STOP')
    window.stopButton.pressed.connect(end)
    window.interationsInput.setEnabled(False)
    pl.ms.x0 = 80
    asyncio.create_task(plStart())

@asyncSlot()
async def end():
    window.ended = True
    window.stopButton.setEnabled(False)

window.mtsButton.pressed.connect(mts)
window.stopButton.pressed.connect(start)

with loop:
    loop.run_forever()
