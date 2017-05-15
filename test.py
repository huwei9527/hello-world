#!/usr/bin/python

import experiment
import sys


def test(a=1, b=2, c=3):
    print a, b, c
    return


print 'Experiment start...'
experiment.test(sys.argv)


class A(object):

    abc = 200

    def __init__(self):
        return

    def __str__(self):
        return "str_string"

    def __getattr__(self, name):
        if (name == 'ssbbsdsf'):
            return 'good'
        else:
            raise AttributeError(self, name)

    @staticmethod
    def testcm(a):
        print a

    def a(self):
        pass

    def bb():
        print 'A bb'
        return None

    def aa(self):
        print 'aa'
        self.bb()
        return


class B(A):

    def __init__(self):
        super(B, self).__init__()
        self.abcd = 0
        return None

    def bb(self):
        print 'B bb'
        return None


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
