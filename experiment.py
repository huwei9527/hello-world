import numpy as np
import matplotlib.pyplot as plt
import best_arm_identification as bai
import data_structure as ds
import time
import data_set as dset


def main():
    """Main"""
    pass


if __name__ == '__main__':
    main()


v = 0.1
size = 4
exp_conf = [
    ('Naive',
     bai.Configuration(size, v, 0.1, 1.0, 0.25)),
    ('SuccessiveElimination',
     bai.Configuration(size, v, 0.1, 1.0, 0.25)),
    ('MedianElimination',
     bai.Configuration(size, v, 0.1, 1.0, 0.25)),
    ('ExponentialGapElimination',
     bai.Configuration(size, v, 0.1, 1.0, 0.25)),
    ('UpperConfidenceBound',
     bai.Configuration(size, v, 0.1, 1.0, 0.25)),
    ('LUCB',
     bai.Configuration(size, v, 0.1, 1.0, 0.25))]


class Experiment(object):
    """Documentation for Experiment

    """
    def __init__(self, data_set):
        super(Experiment, self).__init__()
        self.data_set = data_set
        self._module = __import__('best_arm_identification')
        return None

    def _presult(self):
        return None

    def _result(self, pbd, rt):
        print pbd.h1
        ret = []
        for i in range(0, 1, pbd.conf.size):
            print pbd.total_pulls()
        if rt == pbd.conf.means.argmax():
            ret.append(True)
            print 'GOOD'
        else:
            ret.append(False)
            print 'BAD'
        size = np.zeros(pbd.conf.size, np.int)
        for i in range(0, pbd.conf.size, 1):
            size[i] = pbd._data[i].total_pulls()
        ret.append(sum(size))
        # print size
        return ret

    def run(self, t, alg_id, num_pulls=None):
        pbd = PreBuildData(self.data_set)
        print 'N = %d' % pbd.conf.size
        alg = getattr(self._module, exp_conf[alg_id][0])(
            exp_conf[alg_id][1], pbd.pull)
        ret = alg.run(t, num_pulls)
        self.path = alg.path()
        return self._result(pbd, ret)

    def plot(self, fx, a, b):
        x = np.linspace(a, b, 100)
        plt.plot(x, fx(x))
        plt.show()
        return None


class Experiment1(object):
    """Documentation for Experiment1

    """
    def __init__(self):
        super(Experiment1, self).__init__()
        set_n_size = np.int(np.ceil(3.0 / 0.5))
        self.set_n = np.zeros(set_n_size, np.int)
        for i in range(0, self.set_n.size, 1):
            self.set_n[i] = np.int(np.ceil(np.power(10, 0.5 * (i + 1))))
        print self.set_n
        return None

    def build_data_set(self):
        for el in self.set_n:
            print 'build data set %d' % el
            PreBuildData.one_sparse_data_sets(el)
        return None

    def run_one(self, size):
        data_set = 'one_sparse_' + str(size)
        ex = Experiment(data_set)
        run_rt = np.zeros(2 * len(exp_conf) - 1, np.bool)
        run_pulls = np.zeros(2 * len(exp_conf) - 1, np.int)
        for i in range(0, len(exp_conf) - 1, 1):
            exp_conf[i][1].n = size
            ret = ex.run(1, i)
            [run_rt[2 * i], run_pulls[2 * i]] = ret
            ret = ex.run(0, i)
            [run_rt[2 * i + 1], run_pulls[2 * i + 1]] = ret
        ret = ex.run(1, len(exp_conf) - 1)
        i = len(exp_conf) - 1
        [run_rt[2 * i], run_pulls[2 * i]] = ret
        np.savez('data/' + data_set + '/result.npz',
                 run_rt=run_rt, run_pulls=run_pulls)
        return None

    def load_result(self, size):
        ret = None
        data_set = 'one_sparse_' + str(size)
        with np.load('data/' + data_set + '/result.npz') as d:
            ret = d['run_pulls']
        print ret
        return ret

    def run(self):
        for i in range(0, self.set_n.size - 2, 1):
            self.run_one(self.set_n[i])
        return None

    def _load(self):
        y = np.zeros(
            [self.set_n.size, 2 * len(exp_conf) - 1], np.int)
        for i in range(0, self.set_n.size, 1):
            y[i][:] = self.load_result(self.set_n[i])[:]
        return y

    def plot(self):
        x = np.zeros(self.set_n.size, _float_type)
        y = np.zeros(self.set_n.size, _float_type)
        for i in range(0, self.set_n.size, 1):
            x[i] = np.log10(self.set_n[i])
        yy = self._load()
        # for j in range(0, self.set_n.size, 1):
        fig, ax = plt.subplots()
        for j in range(0, len(exp_conf), 1):
            for i in range(0, self.set_n.size, 1):
                y[i] = np.log10(yy[i][2*j])
            ax.plot(x, y, label=str(j))
        plt.legend()
        plt.show()
        return None


class Experiment2(object):
    """Documentation for Experiment2

    """
    def __init__(self):
        super(Experiment2, self).__init__()
        set_n_size = np.int(np.ceil(3.0 / 0.5))
        self.set_n = np.zeros(set_n_size, np.int)
        for i in range(0, self.set_n.size, 1):
            self.set_n[i] = np.int(np.ceil(np.power(10, 0.5 * (i + 1))))
        print self.set_n
        return None

    def _run(self, alg_id, size):
        PreBuildData.one_sparse_data_sets(size)
        data_set = 'one_sparse_' + str(size)
        ex = Experiment(data_set)
        pbd = PreBuildData(data_set)
        num_pulls = np.int(np.ceil(pbd.h1))
        print num_pulls
        exp_conf[alg_id][1].n = size
        ex.run(1, alg_id, 15 * num_pulls)
        return ex.path

    def run(self, size):
        num = 10
        path = []
        for i in range(0, num, 1):
            path.append(self._run(0, size))
        x = np.arange(5, path[0].size, 1)
        y = np.zeros(x.size, _float_type)
        for i in range(0, x.size, 1):
            cnt = 0
            for j in range(0, num, 1):
                if (path[j][i] == 0):
                    cnt += 1
            y[i] = 1.0 - cnt * 1.0 / num
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.legend()
        plt.show()
        return None


def _fx(x):
    t = x
    d = 0.1
    return (np.sqrt(np.log(4.0 * 20 * t * t) / d) / t)


def fx(x):
    ret = np.zeros(x.size, _float_type)
    for i in range(0, x.size, 1):
        ret[i] = _fx(x[i])
    return ret


def test(argv):
    # PreBuildData.random(20, 0.1, 3, 'abc')
    time_start = time.clock()
    e = Experiment('one_sparse_4')
    e1 = Experiment1()
    e2 = Experiment2()
    # e1.plot()
    e2.run(20)
    # e1.build_data_set()
    # e1.run_one(10)
    # e1.load_result(1000)
    # e1.run()
    # print PreBuildData.one_sparse_means(20)
    # print e.run(1, 5)
    time_stop = time.clock()
    print 'Running time: %f' % (time_stop - time_start)
    return None
