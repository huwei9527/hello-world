import numpy as np
import matplotlib.pyplot as plt
import best_arm_identification as bai
import data_structure as ds
import time


def main():
    """Main"""
    pass


if __name__ == '__main__':
    main()

_float_type = ds._float_type


class NormalData(object):
    """Normal data"""

    def __init__(self, means, variances):
        """Init function"""
        self.means = means.copy()
        self.vars = variances.copy()
        self.data = ds.Data()
        return

    def __str__(self):
        return (("means: %s\n"
                 "vars: %s\n"
                 "n = %d, data_size = %d, max_id = %d") %
                (self.means,
                 self.vars,
                 self.means.size, self.data.size, self.means.argmax()))

    def __getattr__(self, name):
        if (name == 'size'):
            return self.data.size
        elif (name == 'n'):
            return self.means.size
        else:
            raise AttributeError(name)

    def pull(self, arm_id, t):
        assert arm_id < self.n
        # non-central normal distribution: var * n + mu
        ret = np.random.normal(self.means[arm_id], self.vars[arm_id])
        self.data.append(arm_id, ret)
        return ret

    def save(self, fname='NormalData'):
        np.savez(
            fname,
            means=self.means,
            vars=self.vars,
            id=self.data.arm_id_vector(),
            val=self.data.data_vector())
        return

    def load(self, fname='NormalData.npz'):
        with np.load(fname) as d:
            self.means = d['means']
            self.variances = d['vars']
            self.data.set(d['id'], d['val'])

    def hist(self, arm_id):
        a = np.bincount(self.data.arm_id_vector())
        print a
        b = np.zeros(a[arm_id], _float_type)
        iarm = 0
        idata = 0
        size = self.data.size
        while idata < size:
            if self.data.arm_id[idata] == arm_id:
                b[iarm] = self.data.data[idata]
                iarm += 1
            idata += 1
        plt.hist(b, bins=np.ceil(a[arm_id] / 10.0))
        plt.show()


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

    def pull(self, t):
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
    def random(cls, size, data_set=DEFAULT_DIR):
        means = np.random.uniform(0, 1.0, size)
        sigmas = np.random.uniform(0, 1.0, size)
        PreBuildData.build(size, means, sigmas, data_set)
        return None

    def load(self):
        self.conf = Config(self.data_set)
        self.conf.load()
        self._data = []
        for arm_id in range(0, self.conf.size, 1):
            block = DataBlock(arm_id, self.conf)
            block.load()
            self._data.append(block)
        return

    def pull(self, arm_id, t):
        return self._data[arm_id].pull(t)

    def pull_once(self, arm_id):
        return self._data[arm_id].pull_once()


def test():
    # PreBuildData.random(100, 'random')
    pbd = PreBuildData('random')
    pbd.load()
    print pbd.pull(88, 134)
    print pbd.conf[88]
    means_switch = 1
    random_switch = 0
    test_switch = 1
    alg_switch = 0
    if (means_switch == 1):
        means = np.arange(0, 0.99, 0.2, dtype=_float_type)
    elif (means_switch == 2):
        means = np.arange(0, 0.99, 0.1, dtype=_float_type)
    elif (means_switch == 3):
        means = np.arange(0, 0.99, 0.02, dtype=_float_type)
    if (random_switch == 1):
        means = np.random.permutation(means)
    sigma = 0.25
    vars = np.ones(means.size, dtype=_float_type) * sigma
    nd = NormalData(means, vars)
    print nd
    conf = bai.Configuration(nd.n, 0.1, 0.1, 1.0, 0.25)
    print conf
    alg = None
    if (test_switch == 1):
        alg = bai.Naive(conf, nd.pull)
    elif (test_switch == 2):
        alg = bai.SuccessiveElimination(conf, nd.pull)
    elif (test_switch == 3):
        alg = bai.MedianElimination(conf, nd.pull)
    elif (test_switch == 4):
        alg = bai.ExponentialGapElimination(conf, nd.pull)
    elif (test_switch == 5):
        alg = bai.UpperConfidenceBound(conf, nd.pull)
    print 'ALG start...'
    time_start = time.clock()
    ret = -1
    if (alg_switch == 1):
        ret = alg.run()
    elif (alg_switch == 2):
        ret = alg.run_ls()
    time_stop = time.clock()
    print 'ALG end'
    print 'Running time: %f' % (time_stop - time_start)
    if (ret == nd.means.argmax()):
        print '**********(%d)************' % ret
    else:
        print '----------(%d : %d)-------' % (nd.means.argmax(), ret)
    print 'Total arm pulls: %d' % nd.data.size
    return
