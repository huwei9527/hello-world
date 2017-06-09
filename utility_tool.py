import os
import shutil
import logging
from functools import reduce
import blist
import numpy as np


def main():
    '''Main'''
    sl = SortedList().init(5)
    print(sl)
    print(sl.pop())
    print(sl)
    return None


FLOAT_TYPE = np.float
INT_TYPE = np.int
PRECISION = np.finfo(float).resolution


class Log(object):
    """Documentation for Log

    """
    DEFAULT_LEVEL = logging.DEBUG

    def __init__(self):
        super(Log, self).__init__()
        return None

    @classmethod
    def set_level(cls, logger=None, level=logging.INFO):
        if (logger is not None):
            logger.setLevel(level)
        else:
            cls.DEFAULT_LEVEL = level
        return None

    @classmethod
    def debug(cls, logger=None):
        cls.set_level(logger, logging.DEBUG)
        return None

    @classmethod
    def info(cls, logger=None):
        cls.set_level(logger, logging.INFO)
        return None

    @classmethod
    def factory(cls, name):
        logger = logging.getLogger(name)
        logger.setLevel(cls.DEFAULT_LEVEL)
        logger.addHandler(logging.StreamHandler())
        return (logger)


# Log.info()


class Math(object):
    """Documentation for Math

    """
    def __init__(self):
        super(Math, self).__init__()
        return None

    @classmethod
    def is_equal(cls, a, b):
        return (np.isclose(a, b, PRECISION))

    @classmethod
    def power(cls, base, exp):
        return (np.power(base, exp))

    @classmethod
    def divide(cls, a, b):
        return (np.floor_divide(a, b), np.mod(a, b))

    @classmethod
    def compute_h1(cls, means):
        return (reduce(
            lambda x, y: x + 1.0 / (y * y),
            filter(lambda x: not cls.is_equal(x, 0.0), means.max() - means),
            0.0))


class Malloc(object):
    """Documentation for Malloc

    """
    def __init__(self):
        super(Malloc, self).__init__()
        return None

    @classmethod
    def array(cls, size, data_type, init=0):
        return (np.zeros(size, data_type)
                if (init == 0) else np.ones(size, data_type))

    @classmethod
    def floats(cls, size, init=0):
        return (cls.array(size, FLOAT_TYPE, init))

    @classmethod
    def ints(cls, size, init=0):
        return (cls.array(size, INT_TYPE, init))

    @classmethod
    def arange(cls, size, data_type, step=1):
        return (np.arange(0, size, step, data_type))

    @classmethod
    def nrange(cls, size, step=1):
        return (cls.arange(size, INT_TYPE, step))

    @classmethod
    def frange(cls, size, step=1):
        return (cls.arange(size, FLOAT_TYPE, step))


class Random(object):
    """Documentation for Random

    """
    def __init__(self):
        super(Random, self).__init__()
        return None

    @classmethod
    def set_seed(cls, seed):
        np.random.seed(seed)
        return None

    @classmethod
    def normal(cls, mean, variance, size=1):
        return (np.random.normal(mean, variance, size))

    @classmethod
    def uniform(cls, min_value, max_value, size=1):
        return (np.random.uniform(min_value, max_value, size))

    @classmethod
    def random_int(cls, max_value):
        return (np.random.randint(max_value))


class File(object):
    """Documentation for File

    """
    LOGGER = Log.factory('File')

    def __init__(self):
        super(File, self).__init__()
        return None

    @classmethod
    def path(cls, *args):
        return (os.path.join(*args))

    @classmethod
    def is_exist(cls, path):
        return (os.path.exists(path))

    @classmethod
    def mkdir(cls, path):
        cls.LOGGER.debug('Create directory (%s)', path)
        os.mkdir(path)
        return None

    @classmethod
    def delete_directory(cls, path):
        ret = cls.is_exist(path)
        if (ret):
            cls.LOGGER.debug('Delete directory (%s)', path)
            shutil.rmtree(path)
        return (ret)

    @classmethod
    def np_savez(cls, filename, *args, **kargs):
        np.savez(filename, *args, **kargs)
        return None

    @classmethod
    def np_load(cls, filename):
        return (np.load(filename))


class NumpyArrayData(object):
    """Documentation for DataFile

    """
    DEFAULT_FILENAME = 'default'
    DATA_STRS = []
    LOGGER = Log.factory('NPArrayData')

    def __init__(self, data_dir):
        super(NumpyArrayData, self).__init__()
        self._data_dir = data_dir
        self.set_filename(self.DEFAULT_FILENAME)
        self._datas = {}
        return None

    def _before_save(self):
        pass

    def _after_save(self):
        pass

    def _before_load(self):
        pass

    def _after_load(self):
        pass

    def set_filename(self, name):
        self._filename = File.path(self._data_dir, '%s.npz' % name)
        return None

    def save(self):
        self._before_save()
        self._datas.update([(x, getattr(self, x)) for x in self.DATA_STRS])
        self.LOGGER.debug('Save (%s)', self._filename)
        np.savez(self._filename, **self._datas)
        self._after_save()
        return None

    def load(self):
        self._before_load()
        self.LOGGER.debug('Load (%s)', self._filename)
        with np.load(self._filename) as d:
            for el in d:
                setattr(self, el, d[el])
        self._after_load()
        return None


class NumpyList(NumpyArrayData):
    """Documentation for NumpyList

    """
    DEFAULT_FILENAME = 'numpy_list'
    DATA_STRS = ['datas']
    STEP = 10000

    def __init__(self, data_dir='default', data_type=FLOAT_TYPE):
        super(NumpyList, self).__init__(data_dir)
        if (data_type == FLOAT_TYPE):
            self.datas = Malloc.floats(NumpyList.STEP)
        elif (data_type == INT_TYPE):
            self.datas = Malloc.ints(NumpyList.STEP)
        else:
            raise TypeError(data_type)
        self._size = 0
        return None

    def _malloc(self):
        self.data.resize(self.data.size + self.STEP)
        return None

    def __getitem__(self, index):
        return self.datas[index]

    def __getattr__(self, name):
        if (name == 'size'):
            return self._size
        else:
            raise AttributeError(name)
        return None

    def append(self, value):
        if self._data.size <= self._size:
            self._malloc()
        self._datas[self._size] = value
        self._size += 1
        return None


class CachedNumpyList(NumpyList):
    """Documentation for CachedList

    """
    def __init__(self, compute, data_dir='default', data_type=FLOAT_TYPE):
        super(CachedNumpyList, self).__init__(data_dir, data_type)
        self._compute = compute
        return None

    def __getitem__(self, index):
        if (self._size <= index):
            self.append(self._compute(index))
        return self.datas[index]

    def __getattr__(self, name):
        if (name == 'size'):
            return self.size
        else:
            raise AttributeError(name)
        return None

    def start_from_index_one(self):
        self.append(0)
        return None


class IndexList(NumpyArrayData):
    """Documentation for IndexList

    """
    DEFAULT_FILENAME = 'index_list'
    DATA_DIR = ['indexes', 'datas']
    STEP = 10000

    def __init__(self, data_dir='default', data_type=FLOAT_TYPE):
        super(IndexList, self).__init__(data_dir, data_type)
        self.indexes = Malloc.ints(self.STEP)
        self.datas = Malloc.floats(self.index.size)
        self._size = 0
        return None

    def __getattr__(self, name):
        if (name == 'size'):
            return self._size
        else:
            raise AttributeError(name)
        return None

    def __getitem__(self, index):
        return (self.indexes[index], self.datas[index])

    def _malloc(self):
        self.indexes = self.indexes.resize(self.index.size + self.STEP)
        self.datas = self.datas.resize(self.index.size)
        return None

    def append(self, index, value):
        if (self._size == self.indexes.size):
            self._malloc()
        self.indexes[self._size] = index
        self.datas[self._size] = value
        self._size += 1
        return None


class SortedListNode(object):
    """Documentation for SortedListNode

    """
    IS_GREATER = True

    def __init__(self, index, value):
        super(SortedListNode, self).__init__()
        self.index = index
        self.value = value
        self._is_delete = False
        return None

    def __str__(self):
        return ("(%d, %f)" % (self.index, self.value))

    def __gt__(self, other):
        return (not self.__le__(other))

    def __lt__(self, other):
        return (not self.__ge__(other))

    def __ge__(self, other):
        if (self.__eq__(other)):
            return True
        else:
            if (Math.is_equal(self.value, other.value)):
                return (self.index > other.index)
            else:
                return (self.value > other.value)
        return None

    def __le__(self, other):
        if (self.__eq__(other)):
            return True
        else:
            if (Math.is_equal(self.value, other.value)):
                return (self.index < other.index)
            else:
                return (self.value < other.value)
        return None

    def __eq__(self, other):
        return (self.index == other.index)

    def __ne__(self, other):
        return (not self.__eq__(other))

    def delete(self):
        self._is_delete = True
        return None


class SortedList(object):
    """Documentation for SortedList

    """
    def __init__(self):
        super(SortedList, self).__init__()
        self.nodes = blist.sorteddict()
        self.datas = blist.sortedlist()
        self._max_id = 0
        self._except_max_id = 0
        return None

    def __str__(self):
        return ('{0}\n{1}'.format(
            ''.join([str(x) for x in self.nodes.values()]),
            ''.join([str(x) for x in self.datas])))

    def __getitem__(self, index):
        return (self.nodes[index].value)

    def __setitem__(self, index, value):
        self.update(index, value)
        return None

    def __delitem__(self, index):
        node = self.nodes[index]
        del self.datas[node]
        node.delete()
        return None

    def __getattr__(self, name):
        if (name == 'size'):
            return len(self.datas)
        else:
            raise AttributeError(name)

    def _set_max_id(self):
        self._max_id = len(self.datas) - 1
        self._max_except_id = self._max_id - 1
        return None

    def init(self, size):
        self.nodes.update([(x, SortedListNode(x, 0.0)) for x in range(size)])
        '''
        self.nodes[0].value = 0.5
        self.nodes[1].value = 2.5
        self.nodes[2].value = 3.0
        self.nodes[3].value = 2.0
        self.nodes[4].value = 1.0
        '''
        for el in self.nodes.values():
            self.datas.add(el)
        self._set_max_id()
        return (self)

    def argmax_except_arms(self, arm_id):
        node = self.nodes[arm_id]
        self.datas.remove(node)
        max_node = self.datas[self._max_except_id]
        self.datas.add(node)
        return (max_node.index)

    def _max(self):
        return (self.datas[self._max_id])

    def max(self):
        return (self._max().value)

    def argmax(self):
        return (self._max().index)

    def pop(self):
        node = self.datas.pop()
        node.delete()
        return (node.index)

    def update(self, index, value):
        node = self.nodes[index]
        self.datas.remove(node)
        node.value = value
        self.datas.add(node)
        return None


if (__name__ == '__main__'):
    main()
