import rbtree
import numpy as np

FLOAT_TYPE = np.float
PRECISION = np.finfo(float).resolution


def is_float_equal(a, b):
    return (np.isclose(a, b, PRECISION))


def malloc_float(size):
    return np.zeros(size, FLOAT_TYPE)


def malloc_float_one(size):
    return np.ones(size, FLOAT_TYPE)


def malloc_int(size):
    return np.zeros(size, np.int)


def malloc_int_ones(size):
    return np.ones(size, np.int)


def set_random_seed(seed):
    np.random.seed(seed)
    return None


def normal(mean, variance, size):
    return np.random.normal(mean, variance, size)


def random_int(max_val):
    return np.random.randint(max_val)


def path_str(path_1, path_2):
    return (path_1 + '/' + path_2)


def compute_H1(means):
    delta = means.max() - means
    ret = 0.0
    for el in delta:
        if (not is_float_equal(el, 0.0)):
            ret = 1.0 / (el * el)
    return ret


class NPArray(object):
    """Documentation for NPArray

    """
    def __init__(self):
        super(NPArray, self).__init__()
        self._data = None
        return None

    def __str__(self):
        return ('%s' % (self._data))

    def __getattr__(self, name):
        if (name == 'size'):
            return self._data.size
        else:
            raise AttributeError(name)
        return None

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value
        return None

    def save(self, fn):
        if (self._data is not None):
            np.save(fn, self._data)
        else:
            print 'Try to save None array'
        return None

    def load(self, fn):
        self._data = np.load(fn)
        return None


class FloatArray(NPArray):
    """Documentation for FloatArray

    """
    def __init__(self, size=None):
        super(FloatArray, self).__init__()
        if (size is not None):
            self._data = malloc_float(size)
        return None


class IntArray(NPArray):
    """Documentation for IntArray

    """
    def __init__(self, size=None):
        super(IntArray, self).__init__()
        if (size is not None):
            self._data = malloc_int(size)
        return None


class FloatList(object):
    """Documentation for FloatList

    """
    STEP = 10000

    def __init__(self):
        super(FloatList, self).__init__()
        self.STEP = 10000
        self._data = malloc_float(self.STEP)
        self._size = 0
        return None

    def _malloc(self):
        self._data.resize(self._data.size + self.STEP)
        return None

    def __getitem__(self, index):
        return self._data[index]

    def __getattr__(self, name):
        if (name == 'size'):
            return self._size
        else:
            raise AttributeError(name)
        return None

    def append(self, value):
        if self._data.size <= self._size:
            self._malloc()
        self._data[self._size] = value
        self._size += 1
        return None


class CachedList(object):
    """Documentation for CachedList

    """
    def __init__(self, compute):
        super(CachedList, self).__init__()
        self._b = FloatList()
        self._compute = compute
        return None

    def __getitem__(self, index):
        if (self._b.size <= index):
            self._b.append(self._compute(index))
        return self._b[index]

    def __getattr__(self, name):
        if (name == 'size'):
            return self._b.size
        else:
            raise AttributeError(name)
        return None

    def append(self, value):
        self._b.append(value)
        return None

    def start_from_index_one(self):
        self._b.append(0)
        return None


class IndexList(object):
    """Documentation for IndexList

    """
    def __init__(self):
        super(IndexList, self).__init__()
        self.STEP = 10000
        self._index = np.zeros(self.STEP, np.int)
        self._data = np.zeros(self._index.size, FLOAT_TYPE)
        self._size = 0
        return None

    def __getattr__(self, name):
        if (name == 'size'):
            return self._size
        else:
            raise AttributeError(name)
        return None

    def __getitem__(self, index):
        return (self._index[index], self._data[index])

    def _malloc(self):
        self._index = self._index.resize(self._index.size + self.STEP)
        self._data = self._data.resize(self._index.size)
        return None

    def append(self, index, data):
        if (self._size == self._index.size):
            self._malloc()
        self._index[self._size] = index
        self._data[self._size] = data
        self._size += 1
        return None


class SortedListNode(object):
    """Documentation for SortedListNode

    """
    def __init__(self, id, val):
        super(SortedListNode, self).__init__()
        self.id = id
        self.val = val
        self._is_delete = False
        self._big = 1
        self._small = -self._big
        return None

    def __str__(self):
        return ("(%d, %f)" % (self.id, self.val))

    def __cmp__(self, obj):
        if (self.id == obj.id):
            return 0
        elif is_float_equal(self.val, obj.val):
            if (self.id < obj.id):
                return self._small
            else:
                return self._big
        elif self.val > obj.val:
            return self._big
        else:
            return self._small
        return None

    def delete(self):
        self._is_delete = True
        return None


class SortedList(object):
    """Documentation for SortedList

    """
    def __init__(self):
        super(SortedList, self).__init__()
        self.node = []
        self.data = rbtree.rbtree()
        return None

    def __str__(self):
        ret = ""
        for el in self.node:
            ret += el.__str__()
        return ret

    def __getitem__(self, index):
        return self.node[index].val

    def __setitem__(self, index, value):
        del self.data[self.node[index]]
        self.node[index].val = value
        self.data[self.node[index]] = 1
        return None

    def __delitem__(self, index):
        del self.data[self.node[index]]
        self.node[index].delete()
        return None

    def __getattr__(self, name):
        if (name == 'size'):
            return len(self.data)
        else:
            raise AttributeError(name)

    def init(self, val_list):
        i = 0
        for el in val_list:
            self.node.append(SortedListNode(i, el))
            self.data[self.node[i]] = 1
            i += 1
        return None

    def append(self, value):
        n = len(self.node)
        self.node.append(SortedListNode(n, value))
        self.data[self.node[n]] = 1
        return None

    def argmax(self):
        return self.data.max().id

    def argmax_except_arms(self, arm_id):
        del self.data[self.node[arm_id]]
        ret = self.data.max().id
        self.data[self.node[arm_id]] = 1
        return ret

    def max(self):
        return self.data.max().val

    def update(self, id, val):
        del self.data[self.node[id]]
        self.node[id].val = val
        self.data[self.node[id]] = 1
        return None


def test():
    return None
