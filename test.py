#!/usr/bin/python

# import experiment
import sys
import numpy as np


def func1(a, b):
    print 'func1'
    return None


def func2(a, b):
    print 'func2'
    return None


def test(fuc):
    def test_c(a, b):
        print type(fuc)
        return fuc(a, b)
    return test_c


print 'Experiment start...'
# experiment.test(sys.argv)
# import data_set; data_set.test(); sys.exit()


def a():
    print 'AAAA'
    return None


def b():
    print 'BBBB'
    return None

c = a() if False else b()


class A(object):

    ABC = 200

    def __init__(self, arg='AAA'):
        print 'CREATE A'
        return

    def __del__(self):
        return None

    def __str__(self):
        return "str_string"

    def __getattr__(self, name):
        if (name == 'conf'):
            print 'CONF'
            return 'CONF'
        else:
            raise AttributeError(name)
        return None

    @classmethod
    def test_class(cls):
        print cls
        print cls.ABC
        return None

    @staticmethod
    def testcm(a):
        print a

    def a(self):
        print self.ABC
        return None


class B(A):

    ABC = 2001

    def __init__(self, arg='BBB'):
        super(B, self).__init__(arg)
        print self.ABC
        return None

    @classmethod
    def test_classa(cls):
        print cls
        return None


A.ABC = 20202020
a = A()
a.test_class()
print a.__class__.__name__


print 'Over.'
#  print bai.data
#  bai.data.save()
#  bai.data.load()
#  print bai
#  bai.data.hist(3)
    #  return
#
#  ae_timeit = timeit.Timer('time_ae()', 'from __main__ import time_ae')
#  rt = ae_timeit.timeit(1)
#  print 'Running time: ', rt

# plt.figure(figsize=(10, 6), dpi=80)
#  plt.subplot(1, 1, 1)
#  X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
#  C, S = np.cos(X), np.sin(X)
#  plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
#  plt.plot(X, S, color="red", linewidth=2.5, linestyle="-")
#  plt.xlim(X.min()*1.1, X.max()*1.1)
#  plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        #  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
#
#  plt.ylim(C.min()*1.1, C.max()*1.1)
#  plt.yticks([-1, 0, 1],
        #  [r'$-1$', r'$0$', r'$+1$'])
#  ax = plt.gca()
#  ax.spines['right'].set_color('none')
#  ax.spines['top'].set_color('none')
#  ax.xaxis.set_ticks_position('bottom')
#  ax.spines['bottom'].set_position(('data', 0))
#
#  plt.show()
