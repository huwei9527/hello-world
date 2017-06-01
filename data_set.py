import numpy as np
import os
import shutil
import data_structure as ds


def main():
    """Main"""
    return None


if (__name__ == '__main__'):
    main()


def compute_means(means, arms, start_id, size, sums, nums):
    for arm_id in range(start_id, start_id + size, 1):
        pass
    return None


class _Normal(object):
    """Documentation for _Normal

    """
    def __init__(self, mean, variance):
        super(_Normal, self).__init__()
        self._mean = mean
        self._variance = variance
        return None

    def random(self, size):
        return ds.normal(self._mean, self._variance, size)


class _DataFile(object):
    """Documentation for _DataFile

    """
    SUB_FILE_NAME_PREFIX = 'd'
    DEFAULT_FILE_NAME = 'default'

    def __init__(self, data_dir):
        super(_DataFile, self).__init__()
        self._file_name = ds.path_str(
            data_dir, self.DEFAULT_FILE_NAME + '.npz')
        self._datas = None
        return None

    def _sub_file_name(self, data_id):
        return ('%s%d' % (self.SUB_FILE_NAME_PREFIX, data_id))

    def set_file_name(self, name):
        self._file_name = name
        return None

    def save(self):
        func_str = 'np.savez(self._file_name'
        for i in range(0, len(self._datas), 1):
            func_str += (', %s=self._datas[%d]' % (self._sub_file_name(i), i))
        func_str += ')'
        eval(func_str)
        return None

    def load(self):
        with np.load(self._file_name) as d:
            func_str = ('(d[\'%s\']' % (self._sub_file_name(0)))
            for i in range(1, len(d.files), 1):
                func_str += (', d[\'%s\']' % (self._sub_file_name(i)))
            func_str += ')'
            self._datas = eval(func_str)
        return None


class _NormalDataConfig(_DataFile):
    """Documentation for _NormalDataConfig

    """
    DEFAULT_FILE_NAME = 'normal'
    MEANS_ID = 0
    VARIANCES_ID = 1

    def __str__(self):
        return '%s\n%s\n' % (self.means, self.variances)

    def __init__(self, data_dir):
        super(_NormalDataConfig, self).__init__(data_dir)
        return None

    def __getattr__(self, name):
        if (name == 'size'):
            return self._datas[self.MEANS_ID].size
        elif (name == 'means'):
            return self._datas[self.MEANS_ID]
        elif (name == 'variances'):
            return self._datas[self.VARIANCES_ID]
        else:
            raise AttributeError(name)
        return None

    def normal_factory(self, arm_id):
        return _Normal(self.means[arm_id],
                       self.variance[arm_id])

    def h1(self):
        return ds.compute_H1(self.means)

    def init(self, means, variances=None):
        if (variances is not None):
            self._datas = (means.copy(), variances.copy())
        else:
            self._datas = (
                means.copy(), ds.malloc_float_one(self._means.size) * 0.25)
        return None


class _ArmsDataConfig(_DataFile):
    """Documentation for _ArmsDataConfig

    """
    DEFAULT_FILE_NAME = 'arms'
    ARM_SUMS_ID = 0
    ARM_NUMS_ID = 1
    BLOCK_NUMS_ID = 2

    def __init__(self, data_dir):
        super(_ArmsDataConfig, self).__init__(data_dir)
        return None

    def __str__(self):
        return '%s\n%s\n%s' % (self._datas[0], self._datas[1], self._datas[2])

    def __getattr__(self, name):
        if (name == 'arm_sums'):
            return self._datas[self.ARM_SUMS_ID]
        elif (name == 'arm_nums'):
            return self._datas[self.ARM_NUMS_ID]
        elif (name == 'block_nums'):
            return self._datas[self.BLOCK_NUMS_ID]
        else:
            raise AttributeError(name)
        return None

    def __getitem__(self, arm_id):
        return (self._datas[0][arm_id],
                self._datas[1][arm_id],
                self._datas[2][arm_id],)

    def __setitem__(self, arm_id, value):
        (self._datas[0][arm_id],
         self._datas[1][arm_id],
         self._datas[2][arm_id]) = value
        return None

    def init(self, size):
        self._datas = (
            ds.malloc_float(size), ds.malloc_int(size), ds.malloc_int(size))
        return None


class _Config(object):
    """Documentation for _Config

    """
    def __init__(self, data_dir):
        super(_Config, self).__init__()
        self.data_dir = data_dir
        self._normal = _NormalDataConfig(self.data_dir)
        self._arms = _ArmsDataConfig(self.data_dir)
        return None

    def __str__(self):
        return '%s\n%s' % (self._normal, self._arms)

    def __getattr__(self, name):
        if (name == 'size'):
            return self._normal.size
        else:
            raise AttributeError(name)
        return None

    def __getitem__(self, arm_id):
        return (self._arms[arm_id])

    def __setitem__(self, arm_id, value):
        self._arms[arm_id] = value
        return None

    def normal_factory(self, arm_id):
        return (self._normal.normal_factory())

    def save(self):
        self._normal.save()
        self._arms.save()
        return None

    def load(self):
        self._normal.load()
        self._arms.load()
        return None

    def init(self, means, variances):
        self._normal.init(means, variances)
        self._arms.init(self._normal.size)
        return None


class _BlockHead(_DataFile):
    """Documentation for _BlockHead

    """
    DEFAULT_FILE_NAME = 'head'
    ALLOC_SIZE = 10000
    ARMS_ID = 0
    MEANS_ID = 1

    def __init__(self, data_dir):
        super(_BlockHead, self).__init__(data_dir)
        return None

    def __getattr__(self, name):
        if (name == 'arms'):
            return self._datas[self.ARMS_ID]
        elif (name == 'means'):
            return self._datas[self.MEANS_ID]
        else:
            raise AttributeError(name)
        return None

    def init(self, size, normal):
        self._datas = ({}, {})
        for i in range(0, self.data.shape[0], 1):
            self.arms[i] = normal.normal_factory(i).random(self.ALLOC_SIZE)
        return None


class _Block(_DataFile):
    """Documentation for _Block

    """
    ARMS_ID = 0
    MEANS_ID = 1

    def __init__(self, data_dir):
        super(_Block, self).__init__(data_dir)
        return None

    def init(self, size):
        return None


class DataBlock(object):
    """Documentation for DataBlock

    """
    ALLOC_SIZE = 10000
    BLOCK_SIZE = 1000000

    def __init__(self, arm_id, conf):
        super(DataBlock, self).__init__()
        self.data_dir = conf.data_dir

        self._arm_id = arm_id
        self._normal = conf.normal_factory(self._arm_id)
        self._arm_sum = 0.0
        self._arm_num = 0
        self._block_num = 0
        self.set(conf)

        self._block_id = 0
        self._arms = None
        self._means = None
        return None

    def __str__(self):
        return (("arm_id = %d, block_id = %d\n"
                 "means = %s") % (
                     self._arm_id, self._block_id,
                     self._means))

    def __getitem__(self, index):
        return self._means[index]

    def _normal(self, size):
        return np.random.normal(self._mean, self._variance, size)

    def _block_ceil(self, n):
        return np.int(
            np.ceil(n * 1.0 / self._conf.ALLOC_SIZE) * self._conf.ALLOC_SIZE)

    def _filename(self):
        return (ds.path_str(
            self.data_dir,
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
        self._means = np.zeros(arms.size, FLOAT_TYPE)
        self._block_id = 0
        self.sum = 0
        self.size = -1
        self._save(arms, 0)
        return None

    def get(self):
        return (self._arm_sum, self._arm_num, self._block_num)

    def set(self, conf):
        (self._arm_sum, self._arm_num, self._block_num) = conf[self._arm_id]
        return None

    def _raw_load(self):
        with np.load(self._filename()) as d:
            self._means = d['means']
        return None

    def _raw_new_block(self, end_id):
        arms = self._normal(end_id)
        self._means = np.zeros(arms.size, FLOAT_TYPE)
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


class _RandomSeed(_DataFile):
    """Documentation for _RandomSeed

    """
    DEFAULT_FILE_NAME = 'seed'
    SEED_ID = 0

    def __init__(self, data_dir):
        super(_RandomSeed, self).__init__(data_dir)
        return None

    def __getattr__(self, name):
        if (name == 'seed'):
            return self._datas[self.SEED_ID]
        else:
            raise AttributeError(name)
        return None

    def init(self, seed=None):
        if (seed is not None):
            self._datas = (seed, )
        else:
            self._datas = (ds.random_int(65535), )
        return None

    def set_seed(self):
        ds.set_random_seed(self.seed)
        return None


class PreBuildData(object):
    """Documentation for PreBuildData

    """
    def __init__(self, data_dir):
        super(PreBuildData, self).__init__()
        self.data_set = data_dir
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
        means = np.zeros(size, FLOAT_TYPE)
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
    def build(cls, size, means, sigmas, data_dir):
        conf = Config(data_set)
        conf.init(size, means, sigmas)
        for arm_id in range(0, conf.size, 1):
            block = DataBlock(arm_id, conf)
            block.init()
        return None

    @classmethod
    def random(cls, size, gap=0.1, num=1, data_dir=None):
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
    def sample(cls, gap, data_dir):
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


def test():
    a = _BlockHead('.')
    a.init(100)
    return None
