import numpy as np
import data_structure as ds
import os
import shutil
import util


def main():
    """Main"""
    return None


if (__name__ == '__main__'):
    main()


class _DataFile(object):
    """Documentation for _DataFile

    """
    SUB_FILE_NAME_PREFIX = 'd'
    DEFAULT_FILE_NAME = 'default'
    DATA_STRS = []

    def __init__(self, data_dir):
        super(_DataFile, self).__init__()
        self._data_dir = data_dir
        self._set_file_name(self.DEFAULT_FILE_NAME)
        self._datas = {}
        return None

    def _var_str(self, name):
        return ('_%s' % name)

    def _prepare_save(self):
        self._datas.update(
            map(lambda x: (x, getattr(self, self._var_str(x))),
                self.DATA_STRS))
        return None

    def _before_save(self):
        pass

    def _after_load(self):
        pass

    def _set_data(self, name, value):
        setattr(self, name, value)
        # super(_DataFile, self).__setattr__(name, value)
        return None

    def _path_str(self, data_dir, name):
        return (ds.path_str(data_dir, '%s.npz' % name))

    def _set_file_name(self, name):
        self._file_name = self._path_str(self._data_dir, name)
        return None

    def save(self):
        self._before_save()
        self._prepare_save()
        print 'Save file (%s)' % self._file_name
        np.savez(self._file_name, **self._datas)
        return None

    def load(self):
        print 'Load file (%s)' % self._file_name
        with np.load(self._file_name) as d:
            map(lambda x: setattr(self, self._var_str(x), d[x]), d)
        self._after_load()
        return None


class _RandomSeed(_DataFile):
    """Documentation for _RandomSeed

    """
    DEFAULT_FILE_NAME = 'seed'
    DATA_STRS = ['seed']
    MAX_SEED = 65535

    def __init__(self, data_dir):
        super(_RandomSeed, self).__init__(data_dir)
        return None

    def __getattr__(self, name):
        if (name == 'seed'):
            return self._seed
        else:
            raise AttributeError(name)
        return None

    def init(self, seed=None):
        self._seed = (
            seed if seed is not None
            else ds.random_int(_RandomSeed.MAX_SEED))
        return None

    def set_seed(self):
        ds.set_random_seed(self._seed)
        return None


class _NormalDataConfig(_DataFile):
    """Documentation for _NormalDataConfig

    """
    DEFAULT_FILE_NAME = 'normal'
    DATA_STRS = ['means', 'variances']

    def __str__(self):
        return '%s\n%s\n' % (self._means, self._variances)

    def __init__(self, data_dir):
        super(_NormalDataConfig, self).__init__(data_dir)
        return None

    def __getattr__(self, name):
        if (name == 'size'):
            return self._means.size
        else:
            raise AttributeError(name)
        return None

    def random(self, arm_id, size):
        return (ds.normal(self._means[arm_id], self._variances[arm_id], size))

    def h1(self):
        return ds.compute_H1(self._means)

    def init(self, means, variances=None):
        self._means = means.copy()
        self._variances = (
            variances.copy() if variances is not None
            else ds.malloc_float_one(self.size) * 0.25)
        return None


class _ArmsDataConfig(_DataFile):
    """Documentation for _ArmsDataConfig

    """
    DEFAULT_FILE_NAME = 'arms'
    DATA_STRS = ['arm_sums', 'arm_nums', 'block_nums', 'block_sizes']
    SINGLE_BLOCK_SIZE_ID = 0
    MULTI_BLOCK_SIZE_ID = 1

    def __init__(self, data_dir):
        super(_ArmsDataConfig, self).__init__(data_dir)
        return None

    def __str__(self):
        return ('%s\n%s\n%s' % (
            self._arm_sums, self._arm_nums, self._block_nums))

    def __getattr__(self, name):
        if (name == 'arm_sums'):
            return self._arm_sums
        elif (name == 'arm_nums'):
            return self._arm_nums
        elif (name == 'block_nums'):
            return self._block_nums
        elif (name == 'single_block_size'):
            return self._block_sizes[self.SINGLE_BLOCK_SIZE_ID]
        elif (name == 'multi_block_size'):
            return self._block_sizes[self.MULTI_BLOCK_SIZE_ID]
        else:
            raise AttributeError(name)
        return None

    def init(self, size, single_block_size, multi_block_size):
        self._arm_sums = ds.malloc_float(size)
        self._arm_nums = ds.malloc_int(size)
        self._block_nums = ds.malloc_int(size)
        self._block_sizes = ds.malloc_int(2)
        self._block_sizes[self.SINGLE_BLOCK_SIZE_ID] = single_block_size
        self._block_sizes[self.MULTI_BLOCK_SIZE_ID] = multi_block_size
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
        elif (name == 'arm_sums'):
            return self._arms.arm_sums
        elif (name == 'arm_nums'):
            return self._arms.arm_nums
        elif (name == 'block_nums'):
            return self._arms.block_nums
        elif (name == 'single_block_size'):
            return self._arms.single_block_size
        elif (name == 'multi_block_size'):
            return self._arms.multi_block_size
        elif (name == 'h1'):
            return self._normal.compute_H1()
        else:
            raise AttributeError(name)
        return None

    def block_data_factory(self, arm_id):
        return (_BlockData(arm_id, self))

    def multi_block_factory(self):
        return (_MultiBlocks(self, self.multi_block_size))

    def single_block_factory(self, arm_id):
        return (_SingleBlock(self, arm_id, self.single_block_size))

    def data_block_factory(self):
        db = _DataBlock(self)
        db.load()
        return db

    def random(self, arm_id, size):
        return (self._normal.random(arm_id, size))

    def save(self):
        self._normal.save()
        self._arms.save()
        return None

    def load(self):
        self._normal.load()
        self._arms.load()
        return None

    @classmethod
    def init(cls, data_dir, means, variances,
             single_block_size=1000000, multi_block_size=10000):
        config = cls(data_dir)
        config._normal.init(means, variances)
        config._arms.init(config.size, single_block_size, multi_block_size)
        _DataBlock.init(config)
        config.save()
        return config


class _BlockData(object):
    """Documentation for _BlockData

    """
    def __init__(self, arm_id, config):
        super(_BlockData, self).__init__()
        self._arm_id = arm_id
        self._conf = config
        self._curr_id = -1
        self._arms = None
        self._means = None
        return None

    def __str__(self):
        return ('%s\n%s' % (self._arms, self._means))

    def __getattr__(self, name):
        if (name == 'size'):
            return (self._arms.size - self._curr_id - 1)
        elif (name == 'arms'):
            return self._arms
        elif (name == 'means'):
            return self._means
        else:
            raise AttributeError(name)
        return None

    def _get_arm_sum(self):
        return (self._conf.arm_sums[self._arm_id])

    def _get_arm_num(self):
        return (self._conf.arm_nums[self._arm_id])

    def _set_arm_sum(self, value):
        self._conf.arm_sums[self._arm_id] = value
        return None

    def _set_arm_num(self, value):
        self._conf.arm_nums[self._arm_id] = value
        return None

    def _random(self, size):
        # return (ds.malloc_float_one(size) * self._arm_id)
        # return (ds.nrange(size) * (self._arm_id + 1))
        return (self._conf.random(self._arm_id, size))

    def _compute_means(self):
        self._means = ds.malloc_float(self.size)
        sum_ = self._get_arm_sum()
        num = self._get_arm_num()
        for i in xrange(self.size):
            sum_ += self._arms[i]
            num += 1
            self._means[i] = sum_ / num
        self._set_arm_sum(sum_)
        self._set_arm_num(num)
        return None

    def init(self, size):
        self._arms = self._random(size)
        self._compute_means()
        self._curr_id = -1
        return None

    def copy(self, arms, means):
        self._arms = arms
        self._means = means
        return None

    def pull(self, t):
        self._curr_id += t
        return (self._means[self._curr_id])


class _BlockDataFile(_DataFile):
    """Documentation for _BlockDataFile

    """
    def __init__(self, config, block_size):
        super(_BlockDataFile, self).__init__(config.data_dir)
        self._conf = config
        self._block_size = block_size
        return None


class _MultiBlocks(_BlockDataFile):
    """Documentation for _MultiBlocks

    """
    DEFAULT_FILE_NAME = 'head'
    DATA_STRS = ['arms', 'means']

    def __init__(self, config, block_size):
        super(_MultiBlocks, self).__init__(config, block_size)
        self._blocks = {}
        self._blocks.update(
            map(lambda x: (x, self._block_factory(x)),
                xrange(self._conf.size)))
        self._arms = None
        self._means = None
        return None

    def __str__(self):
        str_list = []
        map(
            lambda x: str_list.append('HEAD[%d]\n%s\n' % (x, self._blocks[x])),
            self._blocks)
        return (''.join(str_list))

    def _block_factory(self, arm_id):
        return (self._conf.block_data_factory(arm_id))

    def _before_save(self):
        self._arms = []
        self._means = []
        for i in xrange(len(self._blocks)):
            block = self._blocks[i]
            self._arms.append(block.arms)
            self._means.append(block.means)
        return None

    def _after_load(self):
        arm_id = 0
        for (arms, means) in zip(self._arms, self._means):
            self._blocks[arm_id].copy(arms, means)
            arm_id += 1
        return None

    def init(self):
        for item in self._blocks.itervalues():
            item.init(self._block_size)
        self.save()
        return None

    def size(self, arm_id):
        return (self._blocks[arm_id].size)

    def pull(self, arm_id, t):
        return (self._blocks[arm_id].pull(t))


class _SingleBlock(_BlockDataFile):
    """Documentation for _SingleBlock

    """
    DEFAULT_FILE_NAME = 'block'
    DATA_STRS = ['arms', 'means']

    def __init__(self, config, arm_id, block_size):
        super(_SingleBlock, self).__init__(config, block_size)
        self._arm_id = arm_id
        self._block = self._block_factory()
        self._block_id = 0
        return None

    def __str__(self):
        return ('%s' % self._block)

    def __getattr__(self, name):
        if (name == 'block_id'):
            return self._block_id
        else:
            raise AttributeError(name)
        return None

    def _get_block_num(self):
        return (self._conf.block_nums[self._arm_id])

    def _set_block_num(self, value):
        self._conf.block_nums[self._arm_id] = value
        return None

    def _block_factory(self):
        return (self._conf.block_data_factory(self._arm_id))

    def _change_file_name(self):
        self._set_file_name('%d_%d' % (self._arm_id, self._block_id))
        return None

    def _new(self, num):
        self._block_id = self._get_block_num()
        for i in xrange(num):
            self._block_id += 1
            self._init()
        self._set_block_num(self._block_id)
        return None

    def _load(self, new_block_id):
        self._block_id = new_block_id
        self._change_file_name()
        self.load()
        return None

    def _before_save(self):
        self._change_file_name()
        self._arms = self._block.arms
        self._means = self._block.means
        return None

    def _after_load(self):
        self._block.copy(self._arms, self._means)
        return None

    def _is_empty(self):
        return (self._block_id == 0)

    def _set_first_block(self):
        self._block_id = 0
        self._next_block(1)
        return None

    def _next_block(self, num):
        block_num = self._get_block_num()
        new_block_id = self._block_id + num
        if (new_block_id > block_num):
            self._new(new_block_id - block_num)
        else:
            self._load(new_block_id)
        return None

    def _init(self):
        self._block.init(self._block_size)
        self.save()
        return None

    def pull_once(self):
        if (self._is_empty()):
            self._set_first_block()
        size = self._block.size
        if (size == 0):
            self._next_block(1)
        return (self._block.pull(1))

    def pull(self, t):
        ret = 0.0
        if (self._is_empty()):
            self._set_first_block()
        size = self._block.size
        if (size >= t):
            ret = self._block.pull(t)
        else:
            # self._block.pull(size)
            t -= size
            self._next_block((t / self._block_size) + 1)
            ret = self._block.pull(t % self._block_size)
        return ret


class _DataBlock(object):
    """Documentation for _DataBlock

    """
    def __init__(self, config):
        super(_DataBlock, self).__init__()
        self._conf = config
        self._head = self._head_factory()
        self._blocks = {}
        for arm_id in xrange(self._conf.size):
            self._blocks[arm_id] = self._block_factory(arm_id)
        return None

    def __str__(self):
        str_list = []
        str_list.append(str(self._head))
        for key, item in self._blocks.iteritems():
            str_list.append(
                'DATA[%d] : BLOCK[%d]\n%s\n' % (key, item._block_id, item))
        return (''.join(str_list))

    def _head_factory(self):
        return (self._conf.multi_block_factory())

    def _block_factory(self, arm_id):
        return (self._conf.single_block_factory(arm_id))

    def load(self):
        self._head.load()
        return None

    def pull_once(self, arm_id):
        size = self._head.size(arm_id)
        ret = (self._head.pull(arm_id, 1)
               if (size > 0) else
               self._blocks[arm_id].pull_once())
        return ret

    def pull(self, arm_id, t):
        if (t == 1):
            return self.pull_once(arm_id)
        size = self._head.size(arm_id)
        ret = 0.0
        if (size >= t):
            ret = self._head.pull(arm_id, t)
        else:
            self._head.pull(arm_id, size)
            ret = self._blocks[arm_id].pull(t - size)
        return ret

    @classmethod
    def init(cls, config):
        data = cls(config)
        data._head.init()
        return None


class Means(object):
    """Documentation for Means

    """
    def __init__(self):
        super(Means, self).__init__()
        self._means = None
        return None

    @classmethod
    def one_sparse(cls, size):
        ret = ds.malloc_float(size)
        ret[0] = 0.5
        return ret

    @classmethod
    def alpha_sparse(cls, size, alpha):
        ret = ds.malloc_float(size)
        for i in xrange(size):
            ret[i] = 1.0 - ds.power(i * 1.0 / size, alpha)
        return ret

    @classmethod
    def random(cls, size, gap, core_size,
               max_val=0.99, core_gap=0.1, delta=0.5):
        means = ds.malloc_float(size)
        means[0] = max_val
        means[1] = max_val - gap
        means[2:(core_size + 1)] = ds.uniform(
            max_val - gap - core_gap, max_val - gap, core_size - 1)[:]
        means[(core_size + 1):] = ds.uniform(
            0.0, (max_val - gap) * delta, size - 1 - core_size)[:]
        return means


class PreBuildData(object):
    """Documentation for PreBuildData

    """
    DATA_DIR = 'data'

    def __init__(self, data_set):
        super(PreBuildData, self).__init__()
        self._conf = _Config(PreBuildData.data_dir_str(data_set))
        self._data = None
        return None

    def __str__(self):
        return ('%s\n%s' % (self._conf, self._data))

    def __getattr__(self, name):
        if (name == 'h1'):
            return self._conf.h1
        elif (name == 'data_dir'):
            return self._conf.data_dir
        else:
            raise AttributeError(name)
        return None

    def load(self):
        self._conf.load()
        self._data = self._conf.data_block_factory()
        return

    def save(self):
        self._conf.save()
        return None

    def pull(self, arm_id, t):
        return (self._data.pull(arm_id, t))

    @classmethod
    def build(cls, data_set, means, variances=None,
              single_block_size=1000000, multi_block_size=10000):
        data_dir = cls.data_dir_str(data_set)
        cls.delete_data_set(data_set)
        print 'Create directory (%s)' % data_dir
        os.mkdir(data_dir)
        print 'Start building data set (%s)...' % data_set
        _Config.init(data_dir, means, variances,
                     single_block_size, multi_block_size)
        print 'Success'
        return None

    @classmethod
    def factory(cls, data_dir):
        return (PreBuildData(data_dir))

    @classmethod
    def data_dir_str(cls, data_set):
        return (ds.path_str(cls.DATA_DIR, data_set))

    @classmethod
    def delete_data_set(cls, data_set):
        data_dir = cls.data_dir_str(data_set)
        if (os.path.exists(data_dir)):
            print 'Delete directory (%s)' % data_dir
            shutil.rmtree(data_dir)
        return None


class SparseDataSet(PreBuildData):
    """Documentation for SparseDataSet

    """
    NAME_HEAD = 'spa'

    def __init__(self, size, alpha, is_rebuild=False):
        super(SparseDataSet, self).__init__(
            SparseDataSet.data_set_str(size, alpha))
        self._size = size
        self._alpha = alpha
        if is_rebuild or not (os.path.exists(self.data_dir)):
            SparseDataSet.build(self._size, self._alpha)
        return None

    @classmethod
    def data_set_str(cls, size, alpha):
        return ('{0.NAME_HEAD}{2:.1f}_{1:d}'.format(cls, size, alpha))

    @classmethod
    def build(cls, size, alpha,
              single_block_size=1000000, multi_block_size=10000):
        means = (Means.one_sparse(size)
                 if ds.is_float_equal(alpha, 0.0)
                 else Means.alpha_sparse(size, alpha))
        PreBuildData.build(cls.data_set_str(size, alpha), means,
                           single_block_size=single_block_size,
                           multi_block_size=multi_block_size)
        return None


class RandomDataSet(PreBuildData):
    """Documentation for RandomDataSet

    """
    NAME_HEAD = 'rnd'

    def __init__(self, size, gap=0.1, core_size=1, is_rebuild=False):
        super(RandomDataSet, self).__init__(
            RandomDataSet.data_set_str(size, gap, core_size))
        self._size = size
        self._gap = gap
        self._core_size = core_size
        if is_rebuild or not (os.path.exists(self.data_dir)):
            RandomDataSet.build(self._size, self._gap, self._core_size)
        return None

    @classmethod
    def data_set_str(cls, size, gap, core_size):
        return ('{0.NAME_HEAD}{2:.1f}_{3:d}_{1:d}'.format(
            cls, size, gap, core_size))

    @classmethod
    def build(cls, size, gap=0.1, core_size=1,
              single_block_size=1000000, multi_block_size=10000):
        PreBuildData.build(
            cls.data_set_str(size, gap, core_size),
            Means.random(size, gap, core_size),
            single_block_size=single_block_size,
            multi_block_size=multi_block_size)
        return None


def test():
    print 'DATA_SET: test()'
    # pbd = SparseDataSet(5, 0.3)
    pbd = RandomDataSet(5, is_rebuild=True)
    pbd.load()
    pbd.pull(1, 100)
    pbd.save()
    print pbd
    print 'DATA_SET: test() end'
    return None
