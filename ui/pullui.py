from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,\
     QSlider, QTextEdit, QDial, QProgressBar, QLineEdit
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIntValidator, QDoubleValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np

import sys

app = QApplication(sys.argv)

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100) -> None:
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class PullWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.progressBar = QProgressBar()
        self.progressText = QLabel('0/')
        self.interationsInput = QLineEdit('50')
        self.interationsInput.setValidator(QIntValidator())
        self.interationsInput.setFixedWidth(50)
        self.stopButton = QPushButton('STOP')
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.progressText)
        progressLayout.addWidget(self.interationsInput)
        progressLayout.addWidget(self.stopButton)

        self.powerPlot = MplCanvas()
        self.tensionPlot = MplCanvas()
        self.motionPlot = MplCanvas()
        plotLayout.addWidget(self.powerPlot)
        plotLayout.addWidget(self.tensionPlot)
        plotLayout.addWidget(self.motionPlot)

        self.vSlider = QSlider(Qt.Orientation.Horizontal)
        self.vInput = QLineEdit()
        self.vInput.setFixedWidth(100)
        self.aSlider = QSlider(Qt.Orientation.Horizontal)
        self.aInput = QLineEdit()
        self.aInput.setFixedWidth(100)
        self.TSlider = QSlider(Qt.Orientation.Horizontal)
        self.TInput = QLineEdit()
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
        

        self.wDial = QDial()
        self.wInput = QLineEdit()
        self.PIDButton = QPushButton('PID settings')
        subsettingsLayout.addWidget(QLabel('window'))
        subsettingsLayout.addWidget(self.wDial)
        subsettingsLayout.addWidget(self.wInput)
        subsettingsLayout.addWidget(self.PIDButton)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

window = PullWindow()
window.show()

app.exec()

