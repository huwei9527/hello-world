import numpy as np
import matplotlib.pyplot as plt
import best_arm_identification as bai
import data_structure as ds
import time
import os
import shutil


def main():
    """Main"""
    pass


if __name__ == '__main__':
    main()

_float_type = ds._float_type
DATA_DIR = 'data'
DEFAULT_DIR = 'default'


def data_set_file_name(dir_name, data_set, filename):
    return (dir_name + '/' + data_set + '/' + filename)


class Config(object):
    """Documentation for Config

    """
    def __init__(self, data_set=DEFAULT_DIR):
        super(Config, self).__init__()
        self.DATA_DIR = DATA_DIR
        self.ALLOC_SIZE = 10000
        self.BLOCK_SIZE = 100000

        self.data_set = data_set

        self._filename = (
            data_set_file_name(self.DATA_DIR, data_set, 'conf.npz'))
        self.size = 0
        self.means = None
        self.sigmas = None

        self.mean_sums = None
        self.max_arm_nums = None
        self.max_block_nums = None
        return None

    def __str__(self):
        return (("means = %s\n"
                 "sums = %s\n"
                 "bnums = %s\n"
                 "asums = %s") % (
                     self.means,
                     self.mean_sums,
                     self.max_block_nums,
                     self.max_arm_nums))

    def __getitem__(self, index):
        return self.means[index]

    def save(self):
        np.savez(self._filename,
                 size=self.size,
                 means=self.means,
                 sigmas=self.sigmas,
                 sums=self.mean_sums,
                 mbn=self.max_block_nums,
                 man=self.max_arm_nums)
        return None

    def load(self):
        with np.load(self._filename) as d:
            self.size = d['size']
            self.means = d['means']
            self.sigmas = d['sigmas']
            self.mean_sums = d['sums']
            self.max_block_nums = d['mbn']
            self.max_arm_nums = d['man']
        return None

    def init(self, size, means, sigmas):
        self.size = size
        self.means = means.copy()
        self.sigmas = sigmas.copy()
        self.mean_sums = np.zeros(self.size, _float_type)
        self.max_block_nums = np.ones(self.size, np.int)
        self.max_arm_nums = np.zeros(self.size, np.int)
        self.save()
        return None


class DataBlock(object):
    """Documentation for DataBlock

    """
    def __init__(self, arm_id, conf):
        super(DataBlock, self).__init__()
        self._conf = conf

        self.size = 0

        self._arm_id = arm_id
        self._mean = self._conf.means[arm_id]
        self._sigma = self._conf.sigmas[arm_id]

        self._block_id = 0
        self._means = None

        return None

    def __str__(self):
        return (("arm_id = %d, size = %d, block_id = %d\n"
                 "means = %s") % (
                     self._arm_id, self.size, self._block_id,
                     self._means))

    def __getitem__(self, index):
        return self._means[index]

    def _normal(self, size):
        # return np.arange(
        #    self._arm_id + self._conf.max_arm_nums[self._arm_id],
        #    self._arm_id + self._conf.max_arm_nums[self._arm_id] + size,
        #    1,
        #    dtype=_float_type)
        return np.random.normal(self._mean, self._sigma, size)

    def _block_ceil(self, n):
        return np.int(
            np.ceil(n * 1.0 / self._conf.ALLOC_SIZE) * self._conf.ALLOC_SIZE)

    def _filename(self):
        return (data_set_file_name(
            self._conf.DATA_DIR,
            self._conf.data_set,
            str(self._arm_id) + '_' + str(self._block_id) + '.npz'))

    def _compute_means(self, arms, start_id):
        n = self._conf.max_arm_nums[self._arm_id]
        sum = self._conf.mean_sums[self._arm_id]
        for i in range(start_id, arms.size, 1):
            n += 1
            sum += arms[i]
            self._means[i] = sum / n
        self._conf.mean_sums[self._arm_id] = sum
        self._conf.max_arm_nums[self._arm_id] += arms.size - start_id
        return None

    def _save(self, arms, start_id):
        self._compute_means(arms, start_id)
        np.savez(self._filename(), arms=arms, means=self._means)
        self._conf.save()
        return None

    def init(self):
        arms = self._normal(self._conf.ALLOC_SIZE)
        self._means = np.zeros(arms.size, _float_type)
        self._block_id = 0
        self.sum = 0
        self.size = -1
        self._save(arms, 0)
        return None

    def _raw_load(self):
        with np.load(self._filename()) as d:
            self._means = d['means']
        return None

    def _raw_new_block(self, end_id):
        arms = self._normal(end_id)
        self._means = np.zeros(arms.size, _float_type)
        self._block_id = self._conf.max_block_nums[self._arm_id]
        self._conf.max_block_nums[self._arm_id] += 1
        self._save(arms, 0)
        return None

    def _raw_append_block(self, end_id):
        start_id = self._means.size
        arms_val = self._normal(end_id - start_id)
        with np.load(self._filename()) as d:
            arms = d['arms']
        arms.resize(end_id)
        arms[start_id:end_id] = arms_val[0:(end_id - start_id)]
        self._means.resize(end_id)
        self._save(arms, start_id)
        return None

    def load(self):
        self._block_id = 0
        self.size = -1
        self._raw_load()
        return None

    def _new(self, end_id):
        size = self._block_ceil(end_id)
        return self._raw_new_block(size)

    def _append(self, end_id):
        end_id = self._block_ceil(end_id)
        return self._raw_append_block(end_id)

    def _new_one(self):
        return self._raw_new_block(self._conf.ALLOC_SIZE)

    def _append_one(self):
        return self._raw_append_block(self._means.size + self._conf.ALLOC_SIZE)

    def total_pulls(self):
        return (self._block_id * self._conf.BLOCK_SIZE + self.size + 1)

    def pull(self, t):
        if (t == 1):
            return self.pull_once()
        n = self.size + t
        if (n < self._means.size):
            self.size = n
        else:
            n += self._block_id * self._conf.BLOCK_SIZE
            if (n < self._conf.max_arm_nums[self._arm_id]):
                self._block_id = n / self._conf.BLOCK_SIZE
                self._raw_load()
                self.size = n - self._block_id * self._conf.BLOCK_SIZE
            else:
                last_block_id = self._conf.max_block_nums[self._arm_id] - 1
                if (self._block_id != last_block_id):
                    self._block_id = last_block_id
                    self._raw_load()
                n -= self._block_id * self._conf.BLOCK_SIZE
                if (n < self._conf.BLOCK_SIZE):
                    self._append(n + 1)
                    self.size = n
                else:
                    self._append(self._conf.BLOCK_SIZE)
                    n -= self._conf.BLOCK_SIZE
                    while True:
                        if (n < self._conf.BLOCK_SIZE):
                            break
                        self._new(self._conf.BLOCK_SIZE)
                        n -= self._conf.BLOCK_SIZE
                    self._new(n + 1)
                    self.size = n
        return self._means[self.size]

    def pull_once(self):
        self.size += 1
        if (self.size == self._means.size):
            if (self._means.size == self._conf.BLOCK_SIZE):
                last_block_id = self._conf.max_block_nums[self._arm_id] - 1
                if (self._block_id == last_block_id):
                    self._new_one()
                else:
                    self._block_id += 1
                    self._raw_load()
                self.size = 0
            else:
                self._append_one()
        return self._means[self.size]


class PreBuildData(object):
    """Documentation for PreBuildData

    """
    def __init__(self, data_set=DEFAULT_DIR):
        super(PreBuildData, self).__init__()
        self.data_set = data_set
        self.conf = None
        self._data = None
        self.load()

        max_mean = self.conf.means.max()
        self.h1 = 0.0
        for el in (max_mean - self.conf.means):
            if not (np.isclose(el, 0, ds.precision)):
                self.h1 += 1.0 / (el * el)
        return None

    @classmethod
    def one_sparse_means(cls, size):
        means = np.zeros(size, _float_type)
        means[0] = 0.5
        return means

    @classmethod
    def sparse_means(cls, size, alpha):
        means = np.zeros(size, _float_type)
        for i in range(0, size, 1):
            means[i] = 1.0 - np.power(i * 1.0 / size, alpha)
        return means

    @classmethod
    def one_sparse_data_sets(cls, size):
        print 'Build One sparse data set %d' % size
        data_set = 'one_sparse_' + str(size)
        fn = 'data'
        if not (os.path.exists(fn)):
            os.mkdir(fn)
        fn = fn + '/' + data_set
        if not (os.path.exists(fn)):
            os.mkdir(fn)
        else:
            shutil.rmtree(fn)
            os.mkdir(fn)
        PreBuildData.build(
            size,
            PreBuildData.one_sparse_means(size),
            0.25 * np.ones(size, _float_type),
            data_set)
        print 'Build End %d' % size
        return None

    @classmethod
    def build(cls, size, means, sigmas, data_set=DEFAULT_DIR):
        conf = Config(data_set)
        conf.init(size, means, sigmas)
        for arm_id in range(0, conf.size, 1):
            block = DataBlock(arm_id, conf)
            block.init()
        return None

    @classmethod
    def random(cls, size, gap=0.1, num=1, data_set=DEFAULT_DIR):
        fn = 'data'
        if not (os.path.exists(fn)):
            os.mkdir(fn)
        fn = fn + '/' + data_set
        if not (os.path.exists(fn)):
            os.mkdir(fn)
        else:
            shutil.rmtree(fn)
            os.mkdir(fn)
        means = np.zeros(size, _float_type)
        means[0] = 0.9999
        for i in range(1, num+1, 1):
            means[i] = means[0] - gap - 0.0001 * (i - 1)
        for i in range(num + 1, means.size, 1):
            means[i] = np.random.uniform(0, means[1], 1)
        sigmas = np.random.uniform(0, 1.0, size)
        means = np.random.permutation(means)
        PreBuildData.build(size, means, sigmas, data_set)
        return None

    @classmethod
    def sample(cls, gap, data_set=DEFAULT_DIR):
        means = np.arange(0, 0.99, gap, dtype=_float_type)
        sigmas = np.ones(means.size, _float_type) * 0.25
        PreBuildData.build(means.size, means, sigmas, data_set)
        return None

    @classmethod
    def sample_1(cls, data_set='sample_1'):
        return PreBuildData.sample(0.2, data_set)

    @classmethod
    def sample_2(cls, data_set='sample_2'):
        return PreBuildData.sample(0.1, data_set)

    @classmethod
    def sample_3(cls, data_set='sample_3'):
        return PreBuildData.sample(0.02, data_set)

    def load(self):
        self.conf = Config(self.data_set)
        self.conf.load()
        self._data = []
        for arm_id in range(0, self.conf.size, 1):
            block = DataBlock(arm_id, self.conf)
            block.load()
            self._data.append(block)
        return

    def total_pulls(self):
        ret = 0
        for i in range(0, self.conf.size, 1):
            ret += self._data[i].total_pulls()
        return ret

    def pull(self, arm_id, t):
        return self._data[arm_id].pull(t)

    def pull_once(self, arm_id):
        return self._data[arm_id].pull_once()


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
