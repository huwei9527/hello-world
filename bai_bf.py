import numpy as np
import matplotlib.pyplot as plt
import rbtree


def main():
    """Main"""
    pass


if __name__ == '__main__':
    main()

_float_type = np.float
# The precision of computing float numbers
precision = np.finfo(float).resolution
# Default $\epsilon$
_epsilon = 0.01
# Default $\delta$
_delta = 0.1
# Default $\beta$
_beta = 1
# Default means of the normal random variables
_means = np.arange(0, 1.0, 0.2, dtype=_float_type)
# _means[3] = 0.79
# _means = np.random.permutation(_means)
# Default variance of the normal random variables
_sigma = 0.25
_variances = np.ones(_means.size, dtype=_float_type) * _sigma
_num_pulls = 70
# (1 - _nu) is the default confidence of the algorithm
_nu = 0.1


def _is_zero(float_num):
    """Decide whether the given float number is zero"""
    return (np.fabs(float_num) < precision)


def _is_equal(floata, floatb):
    """Decide whether two floats are equal according to the precision"""
    return _is_zero(floata, floatb)


class Configuration(object):
    """Documentation for Configuration

    """
    def __init__(self, size=_means.size,
                 confidence=_nu, epsilon=_epsilon, beta=_beta, sigma=_sigma):
        super(Configuration, self).__init__()
        self.n = size
        self.v = confidence
        self.e = epsilon
        self.b = beta
        self.s = sigma
        return

    def __str__(self):
        return ("n = %d v = %f e = %f b = %f s = %f" %
                (self.n, self.v, self.e, self.b, self.s))


class FloatList(object):
    """Documentation for FloatList

    """
    _step = 10000
    _type = np.float

    def __init__(self):
        super(FloatList, self).__init__()
        self.data = np.zeros(self._step, self._type)
        self.size = 0
        return

    def malloc(self):
        self.data.resize(self.data.size + self._step)
        return

    def __getitem__(self, index):
        return self.data[index]

    def append(self, value):
        if self.data.size <= self.size:
            self.malloc()
        self.data[self.size] = value
        self.size += 1


class Data(object):
    """Dynamic array.

    Member:
        _step: The step of resizing.
        _type: float type
        data: memory
    """

    def __init__(self):
        super(Data, self).__init__()
        self._type = np.float
        self._step = 10000
        self.data = np.zeros(self._step, self._type)
        self.arm_id = np.zeros(self._step, np.int)

        self.size = 0
        return

    def malloc(self):
        """Increase the memory size of the data."""

        self.data.resize(self.data.size + self._step)
        self.arm_id.resize(self.data.size + self._step)
        return

    def __getitem__(self, index):
        return self.data[index]

    def append(self, value, arm_id):
        if self.data.size <= self.size:
            self.malloc()
        self.data[self.size] = value
        self.arm_id[self.size] = arm_id
        self.size += 1

    def data_vector(self):
        return self.data[0:self.size]

    def arm_id_vector(self):
        return self.arm_id[0:self.size]

    def set(self, data, arm_id):
        self.data = data
        self.arm_id = arm_id
        self.size = data.size
        return


class Arms(object):
    """Documentation for Arms

    """
    def __init__(self, size, pull):
        super(Arms, self).__init__()
        self.n = size
        self.active_arms = np.arange(0, self.n, 1, np.int)
        self.pull = pull
        self.t = np.zeros(self.n, np.int)
        self.sums = np.zeros(self.n, _float_type)
        self.means = np.zeros(self.n, _float_type)

    def __getitem__(self, index):
        return self.active_arms[index]

    def _delete(self, flag):
        self.active_arms = self.active_arms[flag]

    def is_undetermined(self):
        return self.active_arms.size > 1

    def set_active_arms(self, arms):
        self.active_arms = arms.active_arms.copy()
        return

    def delete(self, compare):
        i = 0
        flag = np.ones(self.active_arms.size, np.bool)
        del_flag = False
        while i < self.active_arms.size:
            if compare(self.active_arms[i]):
                flag[i] = False
                del_flag = True
            i += 1
        if del_flag:
            self._delete(flag)
        return

    def pull_one(self, arm_id):
        self.sums[arm_id] += self.pull(arm_id)
        self.t[arm_id] += 1
        self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
        return

    def pull_all(self, t=1):
        while t > 0:
            for arm_id in self.active_arms:
                self.sums[arm_id] += self.pull(arm_id)
                self.t[arm_id] += 1
            t -= 1
        for arm_id in self.active_arms:
            self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
        return

    def init_pull(self):
        self.means = SortedList()
        for i in range(0, self.n, 1):
            self.sums[i] = self.pull(i)
            self.t[i] = 1
            self.means.append(self.sums[i] / self.t[i])
        return

    def compute_one_mean(self, arm_id):
        self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
        return

    def compute_means(self):
        for arm_id in self.active_arms:
            self.compute_one_mean(arm_id)
        return

    def max_active_means_arm_id(self):
        max_id = self.active_arms[0]
        for arm_id in self.active_arms:
            if self.means[arm_id] > self.means[max_id]:
                max_id = arm_id
        return max_id

    def active_means_median(self):
        return np.median(self.means[self.active_arms])

    def active_unique_arm_id(self):
        return self.active_arms[0]


class NormalData(object):
    """Normal data"""

    def __init__(self, means, variances, num_pulls=_num_pulls):
        """Init function"""
        self.means = means.copy()
        self.n = self.means.size
        self.variances = variances.copy()
        self.data = Data()
        return

    def __str__(self):
        return (("means: %s\n"
                 "variances: %s\n"
                 "data_size:%d\n") %
                (self.means,
                 self.variances,
                 self.data.size))

    def pull(self, arm_id):
        """Pull the arm_id-th arm without checking the legality of the index.

        You need to make sure that arm_id is in the range.
        """
        assert arm_id < self.n
        # non-central normal distribution: var * n + mu
        ret = np.random.normal(self.means[arm_id], self.variances[arm_id])
        self.data.append(ret, arm_id)
        return ret

    def save(self, fname='NormalData'):
        np.savez(
            fname,
            means=self.means,
            variances=self.variances,
            data=self.data.data_vector(),
            id=self.data.arm_id_vector())
        return

    def load(self, fname='NormalData.npz'):
        with np.load(fname) as d:
            self.means = d['means']
            self.variances = d['variances']
            self.data.set(d['data'], d['id'])

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


class LilStopCondition(object):
    """Documentation for LilStopCondition

    """
    def __init__(self, conf, arms):
        super(LilStopCondition, self).__init__()
        self.conf = conf
        self._arms = arms
        self._b = FloatList()

        self.e_1 = 1 + self.conf.e
        self._d = (
            np.log(self.e_1) * np.power(
                self.conf.v * self.conf.e / (2 + self.conf.e), 1 / self.e_1))

        self._c1 = 1 + np.sqrt(self.conf.e)
        self._c2 = 2 * self.conf.s * self.conf.s * self.e_1
        self._c3 = 2 * self.conf.n / self._d

        self._ucb = SortedList()

        self.max_mean_id = 0
        self.max_rest_ucb_id = 0

        self.init()
        return

    def _compute_b(self, t):
        return (self._c1 * np.sqrt(
            self._c2 * np.log(self._c3 * np.log(self.e_1 * t + 2)) / t))

    def compute_b(self, t):
        if self._b.size >= t:
            ret = self._b[t - 1]
        else:
            ret = self._compute_b(t)
            self._b.append(ret)
        return ret

    def _find_max_arms(self):
        self.max_mean_id = self._arms.means.argmax()
        self.max_rest_ucb_id = self._ucb.argmax_except_arms(self.max_mean_id)
        return

    def init(self):
        b = self._compute_b(1)
        self._arms.init_pull()
        size = self._arms.means.size()
        for i in range(0, size, 1):
            self._ucb.append(self._arms.means[i] + b)
        return

    def stop_condition(self):
        self._find_max_arms()
        return ((self._arms.means[self.max_mean_id] -
                 self._compute_b(self._arms.t[self.max_mean_id]))
                < self._ucb[self.max_rest_ucb_id])

    def pull_one(self, arm_id):
        self._arms.pull_one(arm_id)
        self._ucb.update(
            arm_id,
            self._arm.means[arm_id] + self.compute_b(self._arms.t[arm_id]))
        return

    def pull_all(self):
        self._arms.pull_all()
        self.update_all()
        return

    def update_all(self):
        for el in self._arms.active_arms:
            self._ucb.update(
                el, self._arms.means[el] + self.compute_b(self._arms.t[el]))
        return


class Cache(object):
    """Algorithm cache, holding temporary data

    Args:
        pull: The hook of function for sampling.
        r: The current round number.
    """
    def __init__(self, conf, pull=None):
        super(Cache, self).__init__()
        self.conf = conf
        self.pull = pull
        self.r = 1

    def compute_time(self, means):
        """Compute the order of time in theory."""
        pass


class Naive(Cache):
    """Naive algorithm.

    """
    def __init__(self, conf, pull):
        super(Naive, self).__init__(conf, pull)
        return

    def run(self):
        t = (4.0 * np.log(2 * self.conf.n / self.conf.v)
             / (self.conf.e * self.conf.e))
        arms = Arms(self.conf.n, self.pull)
        arms.pull_all(t)
        return arms.means.argmax()

    def run_ls(self):
        arms = Arms(self.conf.n, self.pull)
        ls_stop = LilStopCondition(self.conf, arms)
        self.r = 1
        while ls_stop.stop_condition():
            ls_stop.pull_all()
            self.r += 1
            if self.r % 10000 == 0:
                print 'r = %d, b = %f' % (self.r, ls_stop.compute_b(self.r))
        return arms.means.argmax()


class SuccessiveElimination(Cache):
    """Successive elimination algorithm.

    Args:
        c: Control constant greater than 4. (Default 4)
        ref_id: The index of the arm with largest current empirical mean.
    """
    def __init__(self, conf, pull):
        super(SuccessiveElimination, self).__init__(conf, pull)
        self.c = 4

        self.c1 = self.c * self.conf.n / self.conf.v

    def compute_threshold(self):
        """Compute the threshold of the successive elimination algorithm.

        Return:
            The threshold value.
        """
        self.threshold = 2 * np.sqrt(
            np.log(self.c1 * self.r * self.r) / self.r)
        return

    def _compare(self, arm_id):
        return (
            self.max_mean - self.arms.means[arm_id] > self.threshold)

    def _delete_arms(self):
        ref_id = self.arms.max_active_means_arm_id()
        self.max_mean = self.arms.means[ref_id]
        self.compute_threshold()
        self.arms.delete(self._compare)

    def run(self):
        """Run the successive algorithm.

        Return:
            The best arm with probability 1 - confidence
        """
        self.arms = Arms(self.conf.n, self.pull)
        while self.arms.is_undetermined():
            self.arms.pull_all(1)
            self._delete_arms()
            self.r += 1
            if (self.r % 10000 == 0):
                print '(r = %d)' % (self.r)
        return self.arms.active_unique_arm_id()

    def run_ls(self):
        self.conf.v = self.conf.v / 2
        self.arms = Arms(self.conf.n, self.pull)
        ls_stop = LilStopCondition(self.conf, self.arms)
        ls_stop.init()
        self.r = 2
        while ls_stop.stop_condition():
            self.arms.pull_all()
            self._delete_arms()
            ls_stop.update_all()
            self.r += 1
            if (self.r % 10000 == 0):
                print '(r = %d)' % (self.r)
        return self.arms.means.argmax()

    def compute_time(self, means):
        """Compute the worst case running time. (The order of time)
        time = \sum{\ln(n / {self.confidence} / {\delta_i}) / (\delta_i)^2}

        Args:
            means: The expect value of the arms. (Not empirical)
        """
        delta = means.max() - means
        ret = 0
        for el in delta:
            if not _is_zero(el):
                ret += np.log(means.size / self.conf.v / el) / (el * el)
        return ret


class MedianElimination(Cache):
    """Median elimination.

    """
    def __init__(self, conf, pull):
        super(MedianElimination, self).__init__(conf, pull)
        return

    def _compare(self, arm_id):
        return (self.arms.means[arm_id] < self.median)

    def _delete_arms(self):
        self.median = self.arms.active_means_median()
        self.arms.delete(self._compare)
        return

    def run(self, arms=None):
        self.arms = Arms(self.conf.n, self.pull)
        if not (arms is None):
            self.arms.set_active_arms(arms)
        self.r = 1
        epsilon = self.conf.e / 4.0
        delta = self.conf.v / 2.0
        while self.arms.is_undetermined():
            t = np.log(3.0 / delta) / (epsilon * epsilon / 4.0)
            print 't = %d' % t
            self.arms.pull_all(t)
            self._delete_arms()
            epsilon *= 0.75
            delta /= 2.0
            self.r += 1
        return self.arms.active_unique_arm_id()

    def compute_time(self, means=None):
        ret = (
            self.conf.n * np.log(1 / self.conf.v)
            / (self.conf.e * self.conf.e))
        return ret


class ExponentialGapElimination(Cache):
    """Exponential gap elimination algorithm

    """
    def __init__(self, num_arms, confidence, pull):
        super(ExponentialGapElimination, self).__init__(
            num_arms, confidence, pull)
        self.ref_id = 0

    def _compare(self, arm_id):
        return self.arms.means[arm_id] < self.ref_mean

    def _delete_arms(self, epsilon, delta):
        ref_id = self._median_elimination(epsilon / 2.0, delta)
        self.ref_mean = self.arms.means[ref_id] - epsilon
        self.arms.delete(self._compare)
        return

    def _median_elimination(self, epsilon, delta):
        me = MedianElimination(self.n, delta, epsilon, self.pull)
        return me.run(self.arms)

    def run(self):
        self.arms = Arms(self.n, self.pull)
        self.r
        epsilon = 1 / 8.0
        while self.arms.is_undetermined():
            print 'ex round %d' % self.r
            print self.arms.active_arms
            delta = self.confidence / (50.0 * self.r * self.r * self.r)
            t = np.ceil((2.0 / (epsilon * epsilon)) * np.log(2.0 / delta))
            self.arms.pull_all(t)
            self._delete_arms(epsilon, delta)
            epsilon /= 2.0
            self.r += 1
        return self.arms.active_arms[0]

    def compute_time(self, means):
        """Compute the order of running time of exponential gap algorithm.
        time = \sum{(1/\delta_i)^2 * \ln(\ln(1/\delta) / self.confidence)
        Args:
            means: The expect value of each arm
        """
        delta = self._compute_mean_delta(means)
        ret = 0
        for el in delta:
            if not _is_zero(el):
                ret += np.log(np.log(1 / el) / self.confidence) / (el * el)
        return ret


class LilBound(object):
    """Documentation for LilBound

    """
    def __init__(self, epsilon, confidence, beta=1.66, sigma=_sigma):
        super(LilBound, self).__init__()
        self.epsilon = epsilon
        self.confidence = confidence
        self.beta = beta
        self.sigma = sigma
        self.b = FloatList()

        self.e_1 = self.epsilon + 1
        self.c = self._compute_c()
        self.delta = self._compute_delta()
        self.lamda = self._compute_lamda()

        self.c1 = (1 + self.beta) * (1 + np.sqrt(self.epsilon))
        self.c2 = 2 * self.sigma * self.sigma * self.e_1
        return

    def _compute_c(self):
        log_value = 1 / np.log(self.e_1)
        pow_value = np.power(log_value, self.e_1)
        return (2 + self.epsilon) * pow_value / self.epsilon

    def _compute_lamda(self):
        return (2 + self.beta) / self.beta

    def lilucb_stop_condition(self, sum_t, max_t):
        return (max_t < 1 + self.lamda * (sum_t - max_t))

    def _compute_delta(self):
        square_delta = (np.sqrt(self.confidence + 0.25) - 0.5)
        return square_delta * square_delta / self.c

    def _compute_b(self, t):
        return (self.c1 *
                np.sqrt(self.c2 *
                        np.log(np.log(self.e_1*t)/self.delta) / t))

    def compute_b(self, t):
        ret = 0
        if t > self.b.size:
            ret = self._compute_b(t)
            self.b.append(ret)
        else:
            ret = self.b[t - 1]
        return ret

    def compute_time(self, means):
        gama_square = (
            (2 + self.beta) * (1 + np.sqrt(self.epsilon)) * self.sigma)
        gama = (2 * gama_square * gama_square * self.e_1)
        means_delta = np.amax(means) - means
        ret = means.size
        for el in means_delta:
            if not _is_zero(el):
                delta_square = 1 / (el * el)
                ret += 5 * gama * delta_square * np.log(np.e / self.delta)
                log_v = np.log(gama * self.e_1 * delta_square / self.delta)
                if log_v > 1:
                    ret += gama * np.log(2 * log_v) * delta_square
                else:
                    ret += gama * np.log(2) * delta_square
        return ret


class UCB(object):
    """Documentation for UCB

    """
    def __init__(self, size, confidence, epsilon, beta):
        super(UCB, self).__init__()
        self.n = size
        self.confidence = confidence
        self.epsilon = epsilon
        self.beta = beta
        self.ucb_node = []
        for i in range(0, size, 1):
            self.ucb_node.append(UCBNode(0, i))
        self.ucb = rbtree.rbtree()

    def set(self, arm_id, val):
        self.ucb.pop()
        self.ucb_node[arm_id].mean = val
        self.ucb[self.ucb_node[arm_id]] = 1
        return self.ucb.min().arm_id

    def init(self, means):
        i = 0
        for el in means:
            self.ucb_node[i].mean = el
            i += 1
        for el in self.ucb_node:
            self.ucb[el] = 1
        return self.ucb.min().arm_id

    def show(self):
        ll = []
        for el in self.ucb_node:
            ll.append(el.mean)
        print ll
        return


class UpperConfidenceBound(Cache):
    """Upper confidence bound algorithm.
    """
    def __init__(self, num_arms, confidence, pull):
        super(UpperConfidenceBound, self).__init__(num_arms, confidence, pull)
        self.epsilon = 0.01
        self.beta = 1.66
        self.lamda = self.compute_lamda(self.beta)
        self.ucb = np.zeros(self.n, _float_type)
        return

    def run(self):
        lilb = LilBound(self.epsilon, self.confidence, self.beta)
        arms = Arms(self.n, self.pull)
        ucb = UCB(self.n, self.confidence, self.epsilon, self.beta)
        arms.pull_all(1)
        sum_t = self.n
        max_t_arm_id = 1
        means = np.zeros(self.n, _float_type)
        for i in range(0, self.n, 1):
            means[i] = arms.means[i] + lilb.compute_b(1)
        max_arm_id = ucb.init(means)
        while lilb.lilucb_stop_condition(sum_t, arms.t[max_t_arm_id]):
            arms.pull_one(max_arm_id)
            sum_t += 1
            if arms.t[max_t_arm_id] < arms.t[max_arm_id]:
                max_t_arm_id = max_arm_id
            max_arm_id = ucb.set(
                max_arm_id,
                (arms.means[max_arm_id] +
                 lilb.compute_b(arms.t[max_arm_id])))
            self.r += 1
        return max_arm_id

    def run_ls(self):
        LilStopCondition(self.n, self.epsilon)
        return

    def compute_time(self, means):
        lilb = LilBound(self.epsilon, self.confidence, self.beta)
        return lilb.compute_time(means)


class LUCB(Cache):
    """Documentation for LUCB

    """
    def __init__(self, num_arms, confidence, pull):
        super(LUCB, self).__init__(num_arms, confidence, pull)
        return

    def run(self):
        return


def test():
    _means = np.arange(0, 0.99, 0.02, dtype=_float_type)
    _variances = np.ones(_means.size, dtype=_float_type) * _sigma
    nd = NormalData(_means, _variances)
    print nd
    conf = Configuration(_means.size)
    nswitch = 2
    alg = None
    if (nswitch == 1):
        conf.e = 0.1
        alg = Naive(conf, nd.pull)
    elif (nswitch == 2):
        alg = SuccessiveElimination(conf, nd.pull)
    elif (nswitch == 3):
        conf.e = 0.1
        alg = MedianElimination(conf, nd.pull)
    elif (nswitch == 3):
        alg = ExponentialGapElimination(nd.n, _nu, nd.pull)
    elif (nswitch == 5):
        alg = UpperConfidenceBound(nd.n, _nu, nd.pull)
    ret = alg.run_ls()
    if ret == nd.means.argmax():
        print 'Success'
    else:
        print '(%d : %d)' % (nd.means.argmax(), ret)
    print alg.compute_time(nd.means)
    print nd.data.size
    return
