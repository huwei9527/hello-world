#!/usr/bin/python

# import experiment
import sys
import numpy as np


def test(a=1, b=2, c=3):
    print a, b, c
    return a, b, c


print 'Experiment start...'
# experiment.test(sys.argv)
import data_set
data_set.test()


a = [('sdfsf', test, 1), 20, 'sadfaaf']
b = np.ones(5, np.float)
c = np.zeros(5, np.int)
# np.savez('test.npz', a=a, b=b, c=c)
ll = [a, b, c]
print type(ll)
np.save('test.npy', ll)
ll = np.load('test.npy')
print type(ll)
print ll.shape
print type(ll[0])
print ll[0]
print type(ll[0][1])

# tt = tuple(f)
# print type(f)
# print type(f[0][0]), f[0].size
# print type(f[1][0]), f[1].size
# print type(f[2][0]), f[2].size


class A(object):

    ABC = 200

    def __init__(self, arg='AAA'):
        # print self.ABC
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

    @staticmethod
    def testcm(a):
        print a

    def a(self):
        print self.ABC
        return None

    def bb():
        print 'A bb'
        return None

    def aa(self):
        print 'aa'
        self.bb()
        return

    def aaprint(self):
        print self.abc
        return None


class B(A):

    ABC = 2001

    def __init__(self, arg='BBB'):
        super(B, self).__init__(arg)
        print self.ABC
        return None

    def bb(self):
        print 'B bb'
        return None


class C(object):
    def __init__(self):
        return


CC = A
a = CC('adfadfafdadfa')


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
