import sys
from sys import path
import asyncio

path += ['..']
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,\
     QSlider, QTextEdit, QDial, QProgressBar, QLineEdit
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from qasync import QEventLoop, asyncSlot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np


from mylib import Puller

app = QApplication(sys.argv)

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100, label=None) -> None:
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

    def changeLine(self, x, y, label):
        if not label in self.lines:
            self.addPlot(x, y, label=label, legend=(len(self.lines) > 0))
            return
        self.lines[label].set_xdata(x)
        self.lines[label].set_ydata(y)
        self.draw_idle()

class PullWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # pulling parameters
        self.v = 5
        self.a = 5
        self.T0 = 10
        self.w = 100
        self.iterations = 50

        # layouts
        mainLayout = QVBoxLayout()

        progressLayout = QHBoxLayout()
        plotLayout = QHBoxLayout()
        settingsLayout = QHBoxLayout()
        sliderLayout = QVBoxLayout()
        subsettingsLayout = QVBoxLayout()

        mainLayout.addLayout(progressLayout)
        mainLayout.addLayout(plotLayout)
        mainLayout.addLayout(settingsLayout)
        settingsLayout.addLayout(sliderLayout)
        settingsLayout.addLayout(subsettingsLayout)

        # widgets
        self.mtsButton = QPushButton('MTS')
        self.progressBar = QProgressBar()
        self.progressBar.setMaximum(self.iterations)
        self.progressText = QLabel('0/')
        self.interationsInput = QLineEdit(str(self.iterations))
        self.interationsInput.setValidator(QIntValidator())
        self.interationsInput.setFixedWidth(50)
        self.stopButton = QPushButton('START')
        progressLayout.addWidget(self.mtsButton)
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

        self.vSlider = QSlider(Qt.Orientation.Horizontal, minimum=.1, maximum=15, value=self.v)
        self.vSlider.setTickInterval(5)
        self.vSlider.setSingleStep(2)
        self.vInput = QLineEdit(str(self.v))                                                   
        self.vInput.setFixedWidth(100)                                                         
        self.aSlider = QSlider(Qt.Orientation.Horizontal, minimum=.1, maximum=15, value=self.a)
        self.aInput = QLineEdit(str(self.a))                                                   
        self.aInput.setFixedWidth(100)                                                         
        self.TSlider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=200, value=self.T0) 
        self.TInput = QLineEdit(str(self.T0))
        self.TInput.setFixedWidth(100)
        vLayout = QHBoxLayout()
        vLayout.addWidget(QLabel('v '))
        vLayout.addWidget(self.vSlider)
        vLayout.addWidget(self.vInput)
        sliderLayout.addLayout(vLayout)
        aLayout = QHBoxLayout()
        aLayout.addWidget(QLabel('a '))
        aLayout.addWidget(self.aSlider)
        aLayout.addWidget(self.aInput)
        sliderLayout.addLayout(aLayout)
        TLayout = QHBoxLayout()
        TLayout.addWidget(QLabel('T0'))
        TLayout.addWidget(self.TSlider)
        TLayout.addWidget(self.TInput)
        sliderLayout.addLayout(TLayout)
        

        self.wDial = QDial(minimum=50, maximum=10000, value=self.w, singleStep=.5)
        self.wInput = QLineEdit(str(self.w))
        self.PIDButton = QPushButton('PID settings')
        subsettingsLayout.addWidget(QLabel('window'))
        subsettingsLayout.addWidget(self.wDial)
        subsettingsLayout.addWidget(self.wInput)
        subsettingsLayout.addWidget(self.PIDButton)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

        self.interationsInput.textChanged.connect(self.onNewIterations)
        self.vSlider.sliderMoved.connect(self.onNewVSlider)
        self.aSlider.sliderMoved.connect(self.onNewASlider)
        self.TSlider.sliderMoved.connect(self.onNewTSlider)
        self.wDial.sliderMoved.connect(self.onNewWDial)


    def updateAllPlots(self, data):
        if not 'time' in data:
            return
        if 'power' in data:
            self.powerPlot.changeLine(data['time'], data['power'], 'power')
        if 'tension' in data:
            self.tensionPlot.changeLine(data['time'], data['tension'], 'tension')
        if 'motorL' in data:
            self.motionPlot.changeLine(data['time'], data['motorL'], 'motorL') 
        if 'motorR' in data:
            self.motionPlot.changeLine(data['time'], data['motorR'], 'motorR')  
        if 'motorM' in data:
            self.motionPlot.changeLine(data['time'], data['motorM'], 'motorM')  
    
    def onNewVSlider(self, v):
        self.vInput.setText(str(v))
        self.v = v
        print(v)
    def onNewASlider(self, a):
        self.aInput.setText(str(a))
        self.a = a       
    def onNewTSlider(self, T):
        self.TInput.setText(str(T))
        self.T0 = T       
    def onNewWDial(self, w):
        self.wInput.setText(str(w))
        self.w = w
    def onNewIterations(self, num):
        if len(num) == 0:
            return
        self.iterations = int(num)
        self.progressBar.setMaximum(num) 
loop = QEventLoop(app)
asyncio.set_event_loop(loop)

window = PullWindow()
window.show()

pl = Puller()
pl.win = window

@asyncSlot()
async def start():
    await pl.ms.MoveToStart()

window.mtsButton.pressed.connect(start)

with loop:
    loop.run_forever()
