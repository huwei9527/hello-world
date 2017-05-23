import rbtree
import numpy as np

_float_type = np.float
precision = np.finfo(float).resolution


class FloatList(object):
    """Documentation for FloatList

    """
    def __init__(self):
        super(FloatList, self).__init__()
        self._step = 10000
        self._type = _float_type
        self._data = np.zeros(self._step, self._type)
        self._size = 0
        return

    def _malloc(self):
        self._data.resize(self._data.size + self._step)
        return

    def __getitem__(self, index):
        return self._data[index]

    def __getattr__(self, name):
        if (name == 'size'):
            return self._size
        else:
            raise AttributeError(name)
        return

    def append(self, value):
        if self._data.size <= self._size:
            self._malloc()
        self._data[self._size] = value
        self._size += 1
        return


class CachedList(object):
    """Documentation for CachedList

    """
    def __init__(self, compute):
        super(CachedList, self).__init__()
        self._b = FloatList()
        self._compute = compute
        return

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
        self._step = 10000
        self._index = np.zeros(self._step, np.int)
        self._data = np.zeros(self._step, _float_type)
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
        self._index = self._index.resize(self._index.size + self._step)
        self._data = self._data.resize(self._index.size)
        return None

    def append(self, index, data):
        if (self._size == self._index.size):
            self._malloc()
        self._index[self._size] = index
        self._data[self._size] = data
        self._size += 1
        return None


class Data(object):
    """Dynamic array.

    Member:
        _step: The step of resizing.
        _type: float type
        data: memory
    """

    def __init__(self):
        super(Data, self).__init__()
        self._type = _float_type
        self._step = 10000

        self._id = np.zeros(self._step, np.int)
        self._val = np.zeros(self._step, self._type)
        self._size = 0
        return

    def _malloc(self):
        self._val.resize(self._val.size + self._step)
        self._id.resize(self._id.size + self._step)
        return

    def __getitem__(self, index):
        return self.data[index]

    def __getattr__(self, name):
        if (name == 'size'):
            return self._size
        else:
            raise AttributeError(name)
        return

    def append(self, id, val):
        if self._val.size <= self._size:
            self._malloc()
        self._id[self._size] = id
        self._val[self._size] = val
        self._size += 1
        return

    def data_vector(self):
        return self._val[0:self._size]

    def arm_id_vector(self):
        return self._id[0:self._size]

    def set(self, id, val):
        self._id = id
        self._val = val
        self._size = val.size
        return


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
        return

    def __str__(self):
        return ("(%d, %f)" % (self.id, self.val))

    def __cmp__(self, obj):
        if (self.id == obj.id):
            return 0
        elif np.isclose(self.val, obj.val, precision):
            if (self.id < obj.id):
                return self._small
            else:
                return self._big
        elif self.val > obj.val:
            return self._big
        else:
            return self._small
        return

    def delete(self):
        self._is_delete = True
        return


class SortedList(object):
    """Documentation for SortedList

    """
    def __init__(self):
        super(SortedList, self).__init__()
        self.node = []
        self.data = rbtree.rbtree()
        return

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
        return

    def __delitem__(self, index):
        del self.data[self.node[index]]
        self.node[index].delete()
        return

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
        return

    def append(self, value):
        n = len(self.node)
        self.node.append(SortedListNode(n, value))
        self.data[self.node[n]] = 1
        return

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
        return


def test():
    return
