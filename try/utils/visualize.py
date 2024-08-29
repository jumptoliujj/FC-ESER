import os
import time

import numpy as np
#import matplotlib
import torch as t
import visdom

from utils.logger import create_logger

#matplotlib.use('Agg')
#from matplotlib import pyplot as plot

class Visualizer(object):
    def __init__(self, env='try', log=True, **kwargs):
        self.vis = visdom.Visdom('localhost', port=8098, env=env, use_incoming_socket=False, **kwargs)
        self._vis_kw = kwargs

        self.index = {}
        self.log_text = ''

        if log:
            log_file = time.strftime('logs/{}/%Y%m%d_%H%M%S.log'.format(env))
            os.system("mkdir -p logs/{}".format(env))
        else:
            log_file = None
        self.slogger = create_logger(log_file=log_file)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            #print(k)
            #print(v)
            if v is not None:
                self.plot(k, v)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d-%H:%M:%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
