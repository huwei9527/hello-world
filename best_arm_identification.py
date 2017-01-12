import numpy as np
import matplotlib.pyplot as plt


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
_variances = np.ones(_means.size, dtype=_float_type) * 0.25
_num_pulls = 70
# (1 - _nu) is the default confidence of the algorithm
_nu = 0.1
_alloc_step = 10000


def _is_zero(float_num):
    """Decide whether the given float number is zero"""
    return (np.fabs(float_num) < precision)


def _is_equal(floata, floatb):
    """Decide whether two floats are equal according to the precision"""
    return _is_zero(floata, floatb)


class NormalData(object):
    """Normal data"""

    def __init__(self, means, variances, num_pulls=_num_pulls):
        """Init function"""
        self.means = means.copy()
        self.n = self.means.size
        self.variances = variances.copy()
        self.data = np.zeros(_alloc_step, dtype=_float_type)
        self.data_id = np.zeros(self.data.shape, dtype=np.int)
        self.data_max = 0
        return

    def __str__(self):
        return (("means: %s\n"
                 "variances: %s\n"
                 "(data_max/max):(%d/%d)\n") %
                (self.means,
                 self.variances,
                 self.data_max, self.data.size))

    def pull(self, arm_id):
        """Pull the arm_id-th arm without checking the legality of the index.

        You need to make sure that arm_id is in the range.
        """
        assert arm_id < self.n
        if not (self.data_max < self.data.size):
            self.data.resize(self.data.size + _alloc_step)
            self.data_id.resize(self.data.size)
        arm_id = int(arm_id)
        # non-central normal distribution: var * n + mu
        self.data[self.data_max] = np.random.normal(self.means[arm_id],
                                                    self.variances[arm_id])
        # maintain the index which arm the value belongs to
        self.data_id[self.data_max] = arm_id
        self.data_max += 1
        return self.data[self.data_max - 1]

    def save(self, fname='NormalData'):
        np.savez(
            fname,
            means=self.means,
            variances=self.variances,
            data=self.data[:self.data_max],
            id=self.data_id[:self.data_max])
        return

    def load(self, fname='NormalData.npz'):
        with np.load(fname) as d:
            self.means = d['means']
            self.variances = d['variances']
            self.data = d['data']
            self.data_id = d['id']
            self.data_max = self.data.size
            for i in range(self.data_max - 10, self.data_max, 1):
                print '%d %f\n' % (self.data_id[i], self.data[i])

    def hist(self, arm_id):
        a = np.bincount(self.data_id)
        print a
        b = np.zeros(a[arm_id], _float_type)
        iarm = 0
        idata = 0
        size = self.data.size
        while idata < size:
            if self.data_id[idata] == arm_id:
                b[iarm] = self.data[idata]
                iarm += 1
            idata += 1
        plt.hist(b, bins=np.ceil(a[arm_id] / 10.0))
        plt.show()


data = NormalData(_means, _variances)


class Cache(object):
    """Algorithm cache, holding temporary data

    Args:
        n: Number of arms
        means: Empirical means of each arms. (Numpy.narray)
        sums: Total of samples of each arms. (Numpy.narray)
        t: The number of samples for each arms. (Numpy.narray)
        confidence: The confidence of the algorithm. (Default 0.1)
        pull: The hook of function for sampling.
        r: The current round number.
        threshold: Threshold for deleting arms.
    """
    def __init__(self, num_arms, confidence=0.1, pull=None):
        super(Cache, self).__init__()
        self.n = num_arms
        self.means = np.zeros(self.n, _float_type)
        self.sums = np.zeros(self.n, _float_type)
        self.t = np.zeros(self.n, np.int)
        self.ub = np.zeros(self.n, _float_type)
        self.confidence = confidence
        self.pull = pull
        self.r = 1
        self.thresholld = None
        self.bound = np.zeros(_alloc_step, _float_type)
        self.bound_max = 0

    def _pull_one_arm(self, arm_id):
        """Sample arm_id-th arm once.

        Args:
            arm_id: The index of the arm
        """
        self.sums[arm_id] += self.pull(arm_id)
        self.t[arm_id] += 1
        return

    def _pull_arms(self, omiga, t):
        """Sample all the arms in set omiga by times.
        Args:
            omiga: The index set of the arms.(Numpy.narray)
            t: The number of times.
        """
        while t > 0:
            for arm_id in omiga:
                self.sums[arm_id] += self.pull(arm_id)
                self.t[arm_id] += 1
            t -= 1
        return

    def _compute_one_mean(self, arm_id):
        """Compute the empirical mean of the arm_id-th arm in the current.
        Args:
            arm_id: The index of the arm.
        """
        self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
        return

    def _compute_means(self, omiga):
        """Compute the empirical means of the arms in set omiga.

        Args:
            omiga: The set of index of the arms.
        """
        for arm_id in omiga:
            self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]

    def _compute_mean_delta(self, means):
        """Compute differences from the largest value of the means.
        delta_i = max_mean - mean_i
        """
        return np.amax(means) - means

    def compute_time(self, means):
        """Compute the order of time in theory."""
        pass

    def compute_bound(self, n, t, epsilon, delta, sigma=0.25):
        """Compute the bound value."""
        e_plus_one = epsilon + 1
        log_value = np.log(e_plus_one * t + 2)
        loglog_value = np.log(2 * log_value / delta)
        sqrt_value = np.sqrt(2 * sigma * sigma * e_plus_one * loglog_value / t)
        return (1 + np.sqrt(epsilon)) * sqrt_value

    def compute_lamda(self, beta):
        ret = (2 + beta) / beta
        return ret * ret

    def compute_ub(self, arm_id, epsilon, delta):
        self.ub[arm_id] = (
            self.means[arm_id] +
            self.compute_bound(self.n, self.t[arm_id], epsilon, delta))
        return

    def lil_stop_condition(self, lamda):
        sum_t = self.t.sum()
        ret = False
        for i in range(0, self.n, 1):
            if self.t[i] >= 1 + lamda * (sum_t - self.t[i]):
                ret = True
                break
        return ret

    def _find_max_ub(self, arm_id):
        max_value = self.ub[0]
        max_id = 0
        for i in range(1, self.ub.size, 1):
            if i != arm_id and max_value < self.ub[i]:
                max_value = self.ub[i]
                max_id = i
        return max_id

    def ls_stop_condition(self, epsilon, delta):
        ret = False
        arm_i = self.means.argmax()
        arm_j = self._find_max_ub(arm_i)
        bi = self.compute_bound(self.n, self.t[arm_i], epsilon, delta)
        bj = self.compute_bound(self.n, self.t[arm_j], epsilon, delta)
        if self.means[arm_i] - bi > self.means[arm_j] + bj:
            ret = True
        return ret


class SuccessiveElimination(Cache):
    """Successive elimination algorithm.

    Args:
        omiga: The set of the active arms. (Numpy.narray)
        c: Control constant greater than 4. (Default 4)
        ref_id: The index of the arm with largest current empirical mean.
    """
    def __init__(self, num_arms, confidence, pull):
        super(SuccessiveElimination, self).__init__(num_arms, confidence, pull)
        self.omiga = None
        self.c = 4
        self.ref_id = 0

    def compute_threshold(self):
        """Compute the threshold of the successive elimination algorithm.

        Return:
            The threshold value.
        """
        self.threshold = 2 * np.sqrt(
            np.log(self.c * self.r * self.r * self.n) / self.r)
        return self.threshold

    def _find_reference_arm(self):
        for arm_id in self.omiga:
            if self.means[arm_id] > self.means[self.ref_id]:
                self.ref_id = arm_id
        return

    def _delete_arms(self):
        max_mean = self.means[self.ref_id]
        self.compute_threshold()
        omiga_flag = np.ones(self.omiga.size, np.bool)
        i = 0
        del_flag = False
        while i < self.omiga.size:
            if max_mean - self.means[self.omiga[i]] > self.threshold:
                omiga_flag[i] = False
                del_flag = True
            i += 1
        if del_flag:
            self.omiga = self.omiga[omiga_flag]

    def run(self):
        """Run the successive algorithm.

        Return:
            The best arm with probability 1 - confidence
        """
        self.omiga = np.arange(0, self.n, 1, np.int)
        self.r += 1
        while self.omiga.size > 1:
            self._pull_arms(self.omiga, 1)
            self._compute_means(self.omiga)
            self._find_reference_arm()
            self._delete_arms()
            self.r += 1
        return self.omiga[0]

    def compute_time(self, means):
        """Compute the worst case running time. (The order of time)
        time = \sum{\ln(n / {self.confidence} / {\delta_i}) / (\delta_i)^2}

        Args:
            means: The expect value of the arms. (Not empirical)
        """
        delta = self._compute_mean_delta(means)
        ret = 0
        for el in delta:
            if not _is_zero(el):
                ret += np.log(means.size / self.confidence / el) / (el * el)
        return ret


class MedianElimination(Cache):
    """Median elimination.

    """
    def __init__(self, num_arms, epsilon, confidence, pull):
        super(MedianElimination, self).__init__(num_arms, confidence, pull)
        self.omiga = None
        self.epsilon = epsilon

    def _delete_arms(self):
        median = np.median(self.means[self.omiga])
        omiga_flag = np.ones(self.omiga.size, np.bool)
        i = 0
        while i < self.omiga.size:
            if self.means[self.omiga[i]] < median:
                omiga_flag[i] = False
            i += 1
        self.omiga = self.omiga[omiga_flag]

    def run(self, omiga):
        self.omiga = omiga.copy()
        self.r = 1
        epsilon = self.epsilon / 4.0
        delta = self.confidence / 2.0
        while self.omiga.size > 1:
            t = np.log(3.0 / delta) / (epsilon * epsilon / 4.0)
            print 'me-el time: %d' % t
            print 'me-el epsilon %f delta %f' % (epsilon, delta)
            self._pull_arms(self.omiga, t)
            self._compute_means(self.omiga)
            self._delete_arms()
            epsilon *= 0.75
            delta /= 2.0
            self.r += 1
        return self.omiga[0]


class ExponentialGapElimination(Cache):
    """Exponential gap elimination algorithm

    """
    def __init__(self, num_arms, confidence, pull):
        super(ExponentialGapElimination, self).__init__(
            num_arms, confidence, pull)
        self.omiga = None
        self.ref_id = 0

    def _delete_arms(self, epsilon):
        omiga_flag = np.ones(self.omiga.size, np.bool)
        del_flag = False
        i = 0
        ref_mean = self.means[self.ref_id] - epsilon
        while i < self.omiga.size:
            if self.means[self.omiga[i]] < ref_mean:
                omiga_flag[i] = False
                del_flag = True
            i += 1
        if del_flag:
            self.omiga = self.omiga[omiga_flag]

    def _median_elimination(self, epsilon, delta):
        me = MedianElimination(self.n, epsilon, delta, self.pull)
        self.ref_id = me.run(self.omiga)
        return

    def run(self):
        self.omiga = np.arange(0, self.n, 1, np.int)
        self.r = 1
        epsilon = 1 / 8.0
        while self.omiga.size > 1:
            delta = self.confidence / (50 * self.r * self.r * self.r)
            t = np.ceil((2.0 / epsilon / epsilon) * np.log(2.0 / delta))
            print 'ex-gap round: %d' % self.r
            print 'ex-gap time: %d' % t
            print 'ex-gap epsilon %f delta %f' % (epsilon, delta)
            self._pull_arms(self.omiga, t)
            self._compute_means(self.omiga)
            self._median_elimination(epsilon / 2.0, delta)
            self._delete_arms(epsilon)
            epsilon /= 2.0
            self.r += 1
        return self.omiga[0]

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


class Naive(Cache):
    """Naive algorithm.

    """
    def __init__(self, num_arms, confidence, pull):
        super(Naive, self).__init__(num_arms, confidence, pull)
        return

    def run(self, epsilon):
        t = 4.0 * np.log(2 * self.n / self.confidence) / (epsilon * epsilon)
        omiga = np.arange(0, self.n, 1, np.int)
        self._pull_arms(omiga, t)
        self._compute_means(omiga)
        return self.means.argmax()


class UpperConfidenceBound(Cache):
    """Upper confidence bound algorithm.
    """
    def __init__(self, num_arms, confidence, pull):
        super(UpperConfidenceBound, self).__init__(num_arms, confidence, pull)
        self.epsilon = 0.01
        self.beta = 1
        self.lamda = self.compute_lamda(self.beta)
        self.ucb = np.zeros(self.n, _float_type)
        return

    def _compute_ucb_bound(self, arm_id):
        self.ucb[arm_id] = (
            self.means[arm_id] +
            (1 + self.beta) * self.compute_bound(
                self.n, self.t[arm_id], self.epsilon, self.delta))

    def _compute_lil_delta(self, epsilon):
        eplus_one = 1 + epsilon
        log_value = 1 / np.log(eplus_one)
        pow_value = np.power(log_value, eplus_one)
        c = (2 + epsilon) * pow_value / epsilon
        square_delta = (np.sqrt(self.confidence + 0.25) - 0.5)
        self.delta = square_delta * square_delta / c
        return self.delta

    def run(self):
        self.epsilon = 0.01
        self.beta = 1.0
        self.lamda = self.compute_lamda(self.beta)
        self.delta = self._compute_lil_delta(self.epsilon)
        omiga = np.arange(0, self.n, 1, np.int)
        self._pull_arms(omiga, 1)
        self._compute_means(omiga)
        for i in range(0, self.n, 1):
            self._compute_ucb_bound(i)
        while not self.lil_stop_condition(self.lamda):
            arm_id = self.ucb.argmax()
            self._pull_one_arm(arm_id)
            self._compute_one_mean(arm_id)
            self._compute_ucb_bound(arm_id)
        print self.t
        return self.t.argmax()


class LUCB(Cache):
    """Documentation for LUCB

    """
    def __init__(self, num_arms, confidence, pull):
        super(LUCB, self).__init__(num_arms, confidence, pull)
        return

    def run(self):
        return


def test():
    nd = NormalData(_means, _variances)
    print nd.means
    print nd.means.argmax()
    # exponential_gap_elimination(nd.n, nd.pull)
    # print nd.data[:nd.data_max]
    # a = SuccessiveElimination(nd.n, _nu, nd.pull)
    # a = ExponentialGapElimination(nd.n, _nu, nd.pull)
    # a = Naive(nd.n, _nu, nd.pull)
    _nu = 0.1
    a = UpperConfidenceBound(nd.n, _nu, nd.pull)
    print a.run()
    print a.compute_time(nd.means)
    print nd.data_max
    # print nd.data_max / a.compute_time(nd.means)
    return
