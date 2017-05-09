import numpy as np
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


class Experiment(object):
    """Documentation for Experiment

    """
    def __init__(self, data_set):
        super(Experiment, self).__init__()
        self.data_set = data_set
        return None

    def _presult(self):
        return None

    def _result(self, pbd, ret):
        print pbd.h1
        for i in range(0, 1, pbd.conf.size):
            print pbd.total_pulls()
        if ret == pbd.conf.means.argmax():
            print 'GOOD'
        else:
            print 'BAD'
        size = np.zeros(pbd.conf.size, np.int)
        for i in range(0, pbd.conf.size, 1):
            size[i] = pbd._data[i].total_pulls()
        print size
        return None

    def naive(self):
        pbd = PreBuildData(self.data_set)
        conf = bai.Configuration(pbd.conf.size, 0.1, 0.01, 1.0, 0.25)
        self._presult()
        alg = bai.Naive(conf, pbd.pull)
        ret = alg.run()
        self._result(pbd, ret)
        return ret

    def se(self):
        pbd = PreBuildData(self.data_set)
        conf = bai.Configuration(pbd.conf.size, 0.1, 0.01, 1.0, 0.25)
        alg = bai.SuccessiveElimination(conf, pbd.pull)
        ret = alg.run_ls()
        self._result(pbd, ret)
        return ret

    def me(self):
        pbd = PreBuildData(self.data_set)
        conf = bai.Configuration(pbd.conf.size, 0.1, 0.1, 1.0, 0.25)
        alg = bai.MedianElimination(conf, pbd.pull)
        ret = alg.run_ls()
        self._result(pbd, ret)
        return ret

    def ege(self):
        pbd = PreBuildData(self.data_set)
        conf = bai.Configuration(pbd.conf.size, 0.1, 0.1, 1.0, 0.25)
        alg = bai.ExponentialGapElimination(conf, pbd.pull)
        ret = alg.run_ls()
        self._result(pbd, ret)
        return ret

    def ucb(self):
        pbd = PreBuildData(self.data_set)
        conf = bai.Configuration(pbd.conf.size, 0.1, 0.1, 1.0, 0.25)
        alg = bai.UpperConfidenceBound(conf, pbd.pull)
        ret = alg.run_ls()
        self._result(pbd, ret)
        return ret

    def lucb(self):
        pbd = PreBuildData(self.data_set)
        conf = bai.Configuration(pbd.conf.size, 0.1, 0.1, 1.0, 0.25)
        alg = bai.LUCB(conf, pbd.pull)
        ret = alg.run()
        self._result(pbd, ret)
        return ret


def test(argv):
    # PreBuildData.random(20, 0.1, 3, 'n20')
    time_start = time.clock()
    e = Experiment('abc')
    e.naive()
    time_stop = time.clock()
    print 'Running time: %f' % (time_stop - time_start)
    return None
