import utility_tool as ut


ut.Log.debug()
logger = ut.Log.factory(__name__)


def main():
    """Main"""
    single_block_size = 6
    multi_block_size = 3
    logger.info('data_set main():')
    with SparseDataSet(3, 0.3, is_rebuild=True,
                       single_block_size=single_block_size,
                       multi_block_size=multi_block_size) as pbd:
        pbd.pull(1, 10)
        logger.debug('%s', pbd)
        pass
    return None


class _RandomSeed(ut.NumpyArrayData):
    """Documentation for _RandomSeed

    """
    DEFAULT_FILENAME = 'seed'
    DATA_STRS = ['seed']
    MAX_SEED = 65535

    def __init__(self, data_dir):
        super(_RandomSeed, self).__init__(data_dir)
        return None

    def init(self, seed=None):
        self.seed = (
            seed if seed is not None
            else ut.Random.random_int(_RandomSeed.MAX_SEED))
        return None

    def set_seed(self):
        ut.Random.set_seed(self._seed)
        return None


class _NormalDataConfig(ut.NumpyArrayData):
    """Documentation for _NormalDataConfig

    """
    DEFAULT_FILENAME = 'normal'
    DATA_STRS = ['means', 'variances']

    def __str__(self):
        return '%s\n%s\n' % (self.means, self.variances)

    def __init__(self, data_dir):
        super(_NormalDataConfig, self).__init__(data_dir)
        return None

    def __getattr__(self, name):
        if (name == 'size'):
            return self.means.size
        elif (name == 'h1'):
            return ut.Math.compute_h1(self.means)
        else:
            raise AttributeError(name)
        return None

    def random(self, arm_id, size):
        return (ut.Random.normal(
            self.means[arm_id], self.variances[arm_id], size))

    def init(self, means, variances=None):
        self.means = means.copy()
        self.variances = (
            variances.copy() if variances is not None
            else ut.Malloc.floats(self.size, 1) * 0.25)
        return None


class _ArmsDataConfig(ut.NumpyArrayData):
    """Documentation for _ArmsDataConfig

    """
    DEFAULT_FILENAME = 'arms'
    DATA_STRS = ['arm_sums', 'arm_nums', 'block_nums', 'block_sizes']
    SINGLE_BLOCK_SIZE_ID = 0
    MULTI_BLOCK_SIZE_ID = 1

    def __init__(self, data_dir):
        super(_ArmsDataConfig, self).__init__(data_dir)
        return None

    def __str__(self):
        return ('%s\n%s\n%s\n%s' % (
            self.arm_sums, self.arm_nums, self.block_nums, self.block_sizes))

    def __getattr__(self, name):
        if (name == 'single_block_size'):
            return self.block_sizes[self.SINGLE_BLOCK_SIZE_ID]
        elif (name == 'multi_block_size'):
            return self.block_sizes[self.MULTI_BLOCK_SIZE_ID]
        else:
            raise AttributeError(name)
        return None

    def init(self, size, single_block_size, multi_block_size):
        self.arm_sums = ut.Malloc.floats(size)
        self.arm_nums = ut.Malloc.ints(size)
        self.block_nums = ut.Malloc.ints(size)
        self.block_sizes = ut.Malloc.ints(2)
        self.block_sizes[self.SINGLE_BLOCK_SIZE_ID] = single_block_size
        self.block_sizes[self.MULTI_BLOCK_SIZE_ID] = multi_block_size
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
            return self._normal.h1
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
        return (db)

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
        self.arms = None
        self.means = None
        return None

    def __str__(self):
        return ('%s\n%s' % (self.arms, self.means))

    def __getattr__(self, name):
        if (name == 'size'):
            return (self.arms.size - self._curr_id - 1)
        else:
            raise AttributeError(name)
        return None

    def _get_config(self):
        return (self._conf.arm_sums[self._arm_id],
                self._conf.arm_nums[self._arm_id])

    def _set_config(self, sum_, num):
        (self._conf.arm_sums[self._arm_id],
         self._conf.arm_nums[self._arm_id]) = (sum_, num)
        return None

    def _random(self, size):
        # return (ds.malloc_float_one(size) * self._arm_id)
        # return (ds.nrange(size) * (self._arm_id + 1))
        return (self._conf.random(self._arm_id, size))

    def _compute_means(self):
        self.means = ut.Malloc.floats(self.size)
        (sum_, num) = self._get_config()
        for i in range(self.size):
            sum_ += self.arms[i]
            num += 1
            self.means[i] = sum_ / num
        self._set_config(sum_, num)
        return None

    def init(self, size):
        self.arms = self._random(size)
        self._compute_means()
        self._curr_id = -1
        return None

    def copy(self, arms, means):
        (self.arms, self.means) = (arms, means)
        return None

    def pull(self, t):
        self._curr_id += t
        return (self.means[self._curr_id])


class _BlockDataFile(ut.NumpyArrayData):
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
    DEFAULT_FILENAME = 'head'
    DATA_STRS = ['arms', 'means']

    def __init__(self, config, block_size):
        super(_MultiBlocks, self).__init__(config, block_size)
        self._blocks = {}
        self._blocks.update(
            [(x, self._block_factory(x)) for x in range(self._conf.size)])
        return None

    def __str__(self):
        return (''.join(['Head[{0:d}]\n{1}\n'.format(x, y)
                         for x, y in self._blocks.items()]))

    def _block_factory(self, arm_id):
        return (self._conf.block_data_factory(arm_id))

    def _before_save(self):
        (self.arms, self.means) = ([], [])
        for i in range(len(self._blocks)):
            block = self._blocks[i]
            self.arms.append(block.arms)
            self.means.append(block.means)
        return None

    def _after_load(self):
        arm_id = 0
        for (arms, means) in zip(self.arms, self.means):
            self._blocks[arm_id].copy(arms, means)
            arm_id += 1
        return None

    def init(self):
        for item in self._blocks.values():
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
    DEFAULT_FILENAME = 'block'
    DATA_STRS = ['arms', 'means']
    LOGGER = ut.Log.factory('_SingleBlock')

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

    def _get_config(self):
        return (self._conf.block_nums[self._arm_id])

    def _set_config(self, block_num):
        self._conf.block_nums[self._arm_id] = block_num
        return None

    def _block_factory(self):
        return (self._conf.block_data_factory(self._arm_id))

    def _change_filename(self):
        self.set_filename('%d_%d' % (self._arm_id, self._block_id))
        return None

    def _init(self):
        self._block.init(self._block_size)
        self.save()
        return None

    def _new(self, num):
        self._block_id = self._get_config()
        for i in range(num):
            self._block_id += 1
            self._init()
        self._set_config(self._block_id)
        return None

    def _load(self, new_block_id):
        self._block_id = new_block_id
        self.load()
        return None

    def _before_save(self):
        self._change_filename()
        self.arms = self._block.arms
        self.means = self._block.means
        return None

    def _before_load(self):
        self._change_filename()
        return None

    def _after_load(self):
        self._block.copy(self.arms, self.means)
        return None

    def _is_empty(self):
        return (self._block_id == 0)

    def _set_first_block(self):
        self._block_id = 0
        self._next_block(1)
        return None

    def _next_block(self, num):
        self.LOGGER.debug('Next block[%d + %d]', self._block_id, num)
        block_num = self._get_config()
        new_block_id = self._block_id + num
        if (new_block_id > block_num):
            self._new(new_block_id - block_num)
        else:
            self._load(new_block_id)
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
            (step, t) = ut.Math.divide(t, self._block_size)
            self._next_block(step + 1)
            ret = self._block.pull(t)
        return ret


class _DataBlock(object):
    """Documentation for _DataBlock

    """
    def __init__(self, config):
        super(_DataBlock, self).__init__()
        self._conf = config
        # import ipdb; ipdb.set_trace()
        self._head = self._head_factory()
        self._blocks = {}
        for arm_id in range(self._conf.size):
            self._blocks[arm_id] = self._block_factory(arm_id)
        return None

    def __str__(self):
        return (''.join(
            [str(self._head)] + [
                'DATA [{0:d} : {1:d}]\n{2}\n'.format(x, y.block_id, y)
                for (x, y) in self._blocks.items()]))

    def _head_factory(self):
        return (self._conf.multi_block_factory())

    def _block_factory(self, arm_id):
        return (self._conf.single_block_factory(arm_id))

    def load(self):
        self._head.load()
        return None

    def pull_once(self, arm_id):
        size = self._head.size(arm_id)
        return (self._head.pull(arm_id, 1)
                if (size > 0) else self._blocks[arm_id].pull_once())

    def pull(self, arm_id, t):
        if (t == 1):
            return self.pull_once(arm_id)
        size = self._head.size(arm_id)
        if (size >= t):
            ret = self._head.pull(arm_id, t)
        else:
            self._head.pull(arm_id, size)
            ret = self._blocks[arm_id].pull(t - size)
        return (ret)

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
        ret = ut.Malloc.floats(size)
        ret[0] = 0.5
        return ret

    @classmethod
    def alpha_sparse(cls, size, alpha):
        ret = ut.Malloc.floats(size)
        for i in range(size):
            ret[i] = 1.0 - ut.Math.power(i * 1.0 / size, alpha)
        return ret

    @classmethod
    def random(cls, size, gap, core_size,
               max_val=0.99, core_gap=0.1, delta=0.5):
        means = ut.Malloc.floats(size)
        means[0] = max_val
        means[1] = max_val - gap
        means[2:(core_size + 1)] = ut.Random.uniform(
            max_val - gap - core_gap, max_val - gap, core_size - 1)[:]
        means[(core_size + 1):] = ut.Random.uniform(
            0.0, (max_val - gap) * delta, size - 1 - core_size)[:]
        return means


class PreBuildData(object):
    """Documentation for PreBuildData

    """
    DATA_DIR = 'data'
    SINGLE_BLOCK_SIZE = 1000000
    MULTI_BLOCK_SIZE = 10000
    LOGGER = ut.Log.factory('PreBuildData')

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

    def __enter__(self):
        self.load()
        return (self)

    def __exit__(self, type_, value, traceback):
        self.save()
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
              single_block_size=1000000,
              multi_block_size=10000):
        data_dir = cls.data_dir_str(data_set)
        ut.File.delete_directory(data_dir)
        ut.File.mkdir(data_dir)
        cls.LOGGER.info('Building data set (%s)...', data_set)
        _Config.init(data_dir, means, variances,
                     single_block_size, multi_block_size)
        return None

    @classmethod
    def data_dir_str(cls, data_set):
        return (ut.File.path(cls.DATA_DIR, data_set))


class SparseDataSet(PreBuildData):
    """Documentation for SparseDataSet

    """
    NAME_HEAD = 'spa'

    def __init__(self, size, alpha, is_rebuild=False,
                 single_block_size=1000000, multi_block_size=10000):
        super(SparseDataSet, self).__init__(
            SparseDataSet.data_set_str(size, alpha))
        self._size = size
        self._alpha = alpha
        if is_rebuild or not (ut.File.is_exist(self.data_dir)):
            SparseDataSet.build(self._size, self._alpha,
                                single_block_size, multi_block_size)
        return None

    @classmethod
    def data_set_str(cls, size, alpha):
        return ('{0.NAME_HEAD}{2:.1f}_{1:d}'.format(cls, size, alpha))

    @classmethod
    def build(cls, size, alpha,
              single_block_size=1000000,
              multi_block_size=10000):
        means = (Means.one_sparse(size)
                 if ut.Math.is_equal(alpha, 0.0)
                 else Means.alpha_sparse(size, alpha))
        PreBuildData.build(cls.data_set_str(size, alpha), means,
                           single_block_size=single_block_size,
                           multi_block_size=multi_block_size)
        return None


class RandomDataSet(PreBuildData):
    """Documentation for RandomDataSet

    """
    NAME_HEAD = 'rnd'

    def __init__(self, size, gap=0.1, core_size=1, is_rebuild=False,
                 single_block_size=1000000, multi_block_size=10000):
        super(RandomDataSet, self).__init__(
            RandomDataSet.data_set_str(size, gap, core_size))
        self._size = size
        self._gap = gap
        self._core_size = core_size
        if is_rebuild or not (ut.File.is_exist(self.data_dir)):
            RandomDataSet.build(self._size, self._gap, self._core_size,
                                single_block_size, multi_block_size)
        return None

    @classmethod
    def data_set_str(cls, size, gap, core_size):
        return ('{0.NAME_HEAD}{2:.1f}_{3:d}_{1:d}'.format(
            cls, size, gap, core_size))

    @classmethod
    def build(cls, size, gap=0.1, core_size=1,
              single_block_size=1000000,
              multi_block_size=10000):
        PreBuildData.build(
            cls.data_set_str(size, gap, core_size),
            Means.random(size, gap, core_size),
            single_block_size=single_block_size,
            multi_block_size=multi_block_size)
        return None


if (__name__ == '__main__'):
    main()
