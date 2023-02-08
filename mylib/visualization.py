import numpy as np
import pandas as pd
import ipywidgets
from IPython import display
from ipywidgets import widgets
from IPython.display import display
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.layouts import layout


class PlotDisplayer():

    def __init__(self, mainParam='time', **plots):
        self.plotsList = []
        self.plotDict = {}
        self.CreateMaket(mainParam, **plots)

    def CreateMaket(self, mainParam, **plots):
        for plName, plData in plots.items():
            pl = ploter(mainParam=mainParam, data=plData, name=plName)
            self.plotDict[plName] = pl
            self.plotsList += [[pl.plot]]
        self.grid = layout(self.plotsList)

    def Apdate(self, **plots):
        for plName, plData in plots.items():
            if plName == 'for_all':
                data = plData
                for name in self.plotDict.keys():
                    pl = self.plotDict[name]
                    pl.Apdate(data)
            else:
                pl = self.plotDict[plName]
                pl.Apdate(plData)

    def Show(self):
        show(self.grid, notebook_handle=True)

    '''def Set(self, plName, hSize=None, lSize=None):
        pl = self.plotDict[plName]
        print(pl.data)
        if hSize == None:
            hSize = pl.hSize0
        if lSize == None:
            lSize = pl.lSize0
        pl.hSize = hSize
        pl.lSize = lSize
        print(pl.hSize,' w ',pl.lSize)
        pl.aploadData()
        print(pl.hSize,' w ',pl.lSize)
        self.grid = layout(self.plotsList)'''


class ploter():
    colorList = ['red', 'blue', 'green', 'yellow']

    def __init__(self, mainParam, data, name='jalko'):
        self.name = name
        self.mainParam = mainParam
        self.hSize0 = 300
        self.lSize0 = 980
        self.hSize = self.hSize0
        self.lSize = self.lSize0
        # self.lines = np.array([])
        self.data = data
        self.aploadData()

    def aploadData(self):
        self.lines = np.array([])
        # self.plot = figure(plot_width=self.lSize, plot_height=self.hSize, title=self.name)  # , sizing_mode="scale_both"
        self.plot = figure(title=self.name)
        i = 0
        for param in self.data:
            if param == self.mainParam:
                continue
            line = self.plot.line(
                x=self.mainParam,
                y=param,
                source=self.data,
                legend_label=param,
                line_color=self.colorList[i % len(self.colorList)])
            self.lines = np.append(self.lines, line)
            i += 1

    def Apdate(self, data):
        self.data = data
        for line in self.lines:
            line.data_source.data = data


class Slider():

    def __init__(self):
        self.Sl = {}
        self.slBtn = {}
        self.BtnFl = {}
        self.BtnBtn = {}
        self.TtBtn = {}
        self.TtPogr = {}

    def NewTt(self, name, pogr=4):
        self.TtBtn[name] = widgets.Output(layout={'border': '1px solid black'})
        self.TtPogr[name] = pogr

    def ChangeValueTt(self, name, x):
        with self.TtBtn[name]:
            print(name + '= ' +
                  ('{:.' + str(self.TtPogr[name]) + 'f}').format(x))
        self.TtBtn[name].clear_output(wait=True)

    def NewSl(self, name, min=0, max=1, step=0.1, value=0):
        self.slBtn[name] = widgets.FloatSlider(min=min,
                                               max=max,
                                               step=step,
                                               value=value,
                                               description=name,
                                               orientation='vertical')
        self.Sl[name] = lambda: self.slBtn[name].value

    def NewBtn(self, name, *description):
        self.BtnBtn[name] = widgets.Button(description=description[0])
        self.BtnFl[name] = 0
        self.BtnBtn[name].on_click(
            lambda x: self.changeflag(name, *description))

    def Display(self, prin=True):
        lst0 = list(self.TtBtn.values())
        lst1 = list(self.slBtn.values())
        lst2 = list(self.BtnBtn.values())
        l = max(len(lst0), len(lst1), len(lst2))
        grid = ipywidgets.GridspecLayout(3, l)
        i = 0
        for gr in lst0:
            grid[0, i] = gr
            i += 1
        i = 0
        for gr in lst1:
            grid[1, i] = gr
            i += 1
        i = 0
        for gr in lst2:
            grid[2, i] = gr
            i += 1
        if prin:
            b0 = ipywidgets.Box(lst0)
            b1 = ipywidgets.Box(lst1)
            b2 = ipywidgets.Box(lst2)
            display(ipywidgets.VBox([b0, b1, b2]))
        # return grid

    def changeflag(self, name, *description):
        l = len(description)
        i = (self.BtnFl[name] + 1) % l
        self.BtnFl[name] = i
        # print(i,description,description[i])
        self.BtnBtn[name].description = description[i]


class Slider2():

    def __init__(self):
        self.Sl = {}
        self.slBtn = {}
        self.BtnFl = {}
        self.BtnBtn = {}
        self.TtBtn = {}
        self.TtPogr = {}

    
    def NewTt(self, name, pogr=4):
        self.TtBtn[name] = widgets.Output(layout={'border': '1px solid black'})
        self.TtPogr[name] = pogr

    def ChangeValueTt(self, name, x):
        with self.TtBtn[name]:
            print(name + '= ' +
                  ('{:.' + str(self.TtPogr[name]) + 'f}').format(x))
        self.TtBtn[name].clear_output(wait=True)

    def NewSl(self, name, min=0, max=1, step=0.1, value=0):
        self.slBtn[name] = widgets.FloatSlider(min=min,
                                               max=max,
                                               step=step,
                                               value=value,
                                               description=name,
                                               orientation='vertical')
        self.Sl[name] = lambda: self.slBtn[name].value

    def NewBtn(self, name, *description):
        self.BtnBtn[name] = widgets.Button(description=description[0])
        self.BtnFl[name] = 0
        self.BtnBtn[name].on_click(
            lambda x: self.changeflag(name, *description))

    def Display(self, prin=True):
        lst0 = list(self.TtBtn.values())
        lst1 = list(self.slBtn.values())
        lst2 = list(self.BtnBtn.values())
        l = max(len(lst0), len(lst1), len(lst2))
        grid = ipywidgets.GridspecLayout(3, l)
        i = 0
        for gr in lst0:
            grid[0, i] = gr
            i += 1
        i = 0
        for gr in lst1:
            grid[1, i] = gr
            i += 1
        i = 0
        for gr in lst2:
            grid[2, i] = gr
            i += 1
        if prin:
            b0 = ipywidgets.Box(lst0)
            b1 = ipywidgets.Box(lst1)
            b2 = ipywidgets.Box(lst2)
            display(ipywidgets.VBox([b0, b1, b2]))
        # return grid

    def changeflag(self, name, *description):
        l = len(description)
        i = (self.BtnFl[name] + 1) % l
        self.BtnFl[name] = i
        # print(i,description,description[i])
        self.BtnBtn[name].description = description[i]



class Interactives():

    def __init__(self):
        self.dict = {}

    class element():

        def __init__(self):
            self.dict = {}

        def New(self, name, value, texn):
            self.dict[name] = (value, texn)

        def GetValue(self, name):
            return self.dict[name][0]

        def GetTexn(self, name):
            return self.dict[name][1]

        def ChangeValue(self, name, value):
            texn = self.GetTexn(name)
            self.dict[name] = (value, texn)