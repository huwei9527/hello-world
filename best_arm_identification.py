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
_means[3] = 0.79
# _means = np.random.permutation(_means)
# Default variance of the normal random variables
_variances = np.ones(_means.size, dtype=_float_type) * 0.25
_num_pulls = 70
# (1 - _nu) is the default confidence of the algorithm
_nu = 0.1


def _is_zero(float_num):
    """Decide whether the given float number is zero"""
    return (np.fabs(float_num) < precision)


def _is_equal(floata, floatb):
    """Decide whether two floats are equal accroding to the precision"""
    return _is_zero(floata, floatb)


def _compute_successive_elimination_threshold(t, n, c=4, delta=_nu):
    """Compute threshold of the successive elimination.
    $\alpha=\ln(c*n*t^2/\delta)/t$
    """
    ret = np.sqrt(np.log(c * t * t * n) / t)
    return ret


def _compute_h_1(means):
    """Compute H_1 value
    $H_1 = \sigma_{i}(\mu_i-\mu_{i_*})^{-2}$
    """
    means_max = means.max()
    result_sum = 0.0
    for x in means:
        x = x - means_max
        if not _is_zero(x):
            result_sum += x**(-2)
    return result_sum


def _compute_u(t, delta=_delta, epsilon=_epsilon):
    """Compute the parameter U."""
    ep_plus_one = epsilon + 1
    log_v = np.log10(ep_plus_one * t + 2)
    loglog_v = np.log10(log_v / delta)
    return (1 + np.sqrt(epsilon)) * np.sqrt(ep_plus_one * loglog_v / (2 * t))


class NormalData(object):
    """Normal data"""

    def __init__(self, means, variances, num_pulls=_num_pulls):
        """Init function"""
        self.means = means.copy()
        self.n = self.means.size
        self.variances = variances.copy()
        self._mem_step = int(num_pulls * _compute_h_1(self.means))
        self.data = np.zeros(self._mem_step, dtype=_float_type)
        self.data_id = np.zeros(self.data.shape, dtype=np.int)
        self.data_max = 0
        return

    def __str__(self):
        return (("means: %s\n"
                 "variances: %s\n"
                 "H1: %f\n"
                 "(data_max/max):(%d/%d)\n") %
                (self.means,
                 self.variances,
                 _compute_h_1(self.means),
                 self.data_max, self.data.size))

    def pull(self, arm_id):
        """Pull the arm_id-th arm without checking the legality of the index.

        You need to make sure that arm_id is in the range.
        """
        assert arm_id < self.n
        if not (self.data_max < self.data.size):
            self.data.resize(self.data.size + self._mem_step)
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


class AlgCache:
    """Cache useful parameters used during the computing."""

    def __init__(self, num_arms, pull_hook=None, conf_hook=None):
        """Init function"""
        self.n = num_arms
        self.means = np.zeros(self.n, _float_type)
        self.sums = np.zeros(self.n, _float_type)

        self.max_mean_id = 0
        self.ref_id = 0
        self.conf = np.zeros(self.n, _float_type)
        self.t = np.zeros(self.n, np.int)
        self.h = 0
        self.l = 1
        assert not (pull_hook is None)
        self.pull_hook = pull_hook
        self.conf_hook = conf_hook
        return

    def __str__(self):
        """format printing"""
        return "measns_exp: %s\nconf: %s\nt: %s\n(h, l): (%d, %d)" % (
            self.means_exp, self.conf, self.t, self.h, self.l)

    def _find_max_expirical_mean_id(self):
        return self.means_exp.argmax()

    def _find_max_confidence_id(self):
        old_conf = self.conf[self.h]
        self.conf[self.h] = self.conf.min() - 1.0
        ret_max_conf_id = self.conf.argmax()
        self.conf[self.h] = old_conf
        return ret_max_conf_id

    def _pull_arms(self, t, omiga):
        while t > 0:
            for arm_id in omiga:
                self.sums[arm_id] += self.pull_hook(arm_id)
                self.t[arm_id] += 1
            t -= 1

    def _successive_elimination_delete_arms(self, t, omiga, delta):
        max_mean = self.means[self.ref_id]
        threshold = (
            2 * _compute_successive_elimination_threshold(t, self.n, 4, delta))
        omiga_flag = np.ones(omiga.size, np.bool)
        i = 0
        del_flag = False
        while i < omiga.size:
            if max_mean - self.means[omiga[i]] > threshold:
                omiga_flag[i] = False
                del_flag = True
            i += 1
        if del_flag:
            return omiga[omiga_flag]
        else:
            return omiga

    def successive_elimination_pull(self, t, omiga, delta):
        self._pull_arms(1, omiga)
        for arm_id in omiga:
            self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
            if self.means[arm_id] > self.means[self.ref_id]:
                self.ref_id = arm_id
        return self._successive_elimination_delete_arms(t, omiga, delta)

    def _median_elimination_pull(self, omiga, epsilon, delta):
        t = np.log(3.0 / delta) / (epsilon * epsilon / 4.0)
        self._pull_arms(t, omiga)
        for arm_id in omiga:
            self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
        median = np.median(self.means[omiga])
        omiga_flag = np.ones(omiga.size, np.bool)
        i = 0
        while i < omiga.size:
            if self.means[omiga[i]] < median:
                omiga_flag[i] = False
            i += 1
        return omiga[omiga_flag]

    def median_elimination(self, omiga, epsilon, delta):
        """Median elimination algorithm"""
        epsilon_r = epsilon / 4.0
        delta_r = delta / 2.0
        omiga_r = omiga.copy()
        dcache = AlgCache(self.n, self.pull_hook)
        while omiga_r.size > 1:
            # print (
            #    'median elimination:''epsilon %f ' 'delta %f') % (
            #        epsilon_r, delta_r)
            # print omiga_r
            omiga_r = dcache._median_elimination_pull(
                omiga_r, epsilon_r, delta_r)
            epsilon_r *= 0.75
            delta_r /= 2.0
        return omiga_r[0]

    def _exponential_gap_elimination_delete_arms(self, omiga, epsilon):
        omiga_flag = np.ones(omiga.size, np.bool)
        del_flag = False
        i = 0
        ref_mean = self.means[self.ref_id]
        while i < omiga.size:
            if self.means[omiga[i]] < ref_mean - epsilon:
                omiga_flag[i] = False
                del_flag = True
            i += 1
        if del_flag:
            return omiga[omiga_flag]
        else:
            return omiga

    def exponential_gap_elimination_pull(self, epsilon, delta, omiga):
        t = np.ceil((2.0 / epsilon / epsilon) * np.log(2.0 / delta))
        self._pull_arms(t, omiga)
        for arm_id in omiga:
            self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
        # print 'pull %d times' % t
        self.ref_id = self.median_elimination(omiga, epsilon / 0.2, delta)
        return self._exponential_gap_elimination_delete_arms(omiga, epsilon)

    def pull(self, arm_id):
        """Pull the arm_id-th arm and maintain the parameters."""
        assert arm_id >= 0 and arm_id < self.n
        out_val = self.pull_hook(arm_id)
        frac = self.t[arm_id] / (self.t[arm_id] + 1.0)
        old_mean = self.means_exp[arm_id]
        self.means_exp[arm_id] = (frac * old_mean + (1 - frac) * out_val)
        self.t[arm_id] += 1
        # compute h = arg max {\mu}
        # compute l = arg max_{i\h} {\mu + c}
        if self.conf_hook is None:
            return
        self.conf[arm_id] = (
            self.means_exp[arm_id] + self.conf_hook(self.t[arm_id]))
        self.h = self._find_max_expirical_mean_id()
        self.l = self._find_max_confidence_id()
        return


def _ae_find_upper_confidence_arm(data_cache, omiga):
    """Find the arm with with max confidence according to the mean value.

    Note that means are usually experianced value.
    """
    return data_cache.conf[omiga].argmax()


def _ae_delete_certian_small_arms(data_cache, omiga, k):
    """Delete small arms accroding to $\mu_a - c < \mu_i + c$"""
    cnt = 0
    c = data_cache.conf_hook(k)
    ref_arm = _ae_find_upper_confidence_arm(data_cache, omiga)
    ref_value = data_cache.means_exp[omiga[ref_arm]] - c
    for i in range(0, omiga.size, 1):
        if not (ref_value < data_cache.means_exp[omiga[i]] + c):
            omiga[i] = -1
            cnt += 1
    if cnt == 0:
        return omiga
    else:
        ret_omiga = np.zeros(omiga.size - cnt, np.int)
        ret_index = 0
        for i in range(0, omiga.size, 1):
            if omiga[i] >= 0:
                ret_omiga[ret_index] = omiga[i]
                ret_index += 1
        return ret_omiga


class AEConfidence:
    """Compute the confidence of the action elimination"""

    def __init__(self, delta=_delta, epsilon=_epsilon):
        self.delta = _delta
        self.epsilon = _epsilon

    def compute(self, t):
        return (2 * _compute_u(t, self.delta, self.epsilon))

    def compute_positive_rate(self):
        ep_plus_one = 1 + self.epsilon
        log_v = np.log10(ep_plus_one)
        tmp = np.power(1 / log_v, ep_plus_one)
        tmp = 1 - ((2 + self.epsilon) * 2 * tmp * self.delta / self.epsilon)
        return tmp


def action_elimination(num_arms, pull=None, delta=_delta, epsilon=_epsilon):
    """Nonadaptive LS algorithm"""
    conf_hook = AEConfidence(delta / num_arms, epsilon)
    print conf_hook.compute_positive_rate()
    dcache = AlgCache(num_arms, pull, conf_hook.compute)
    k = 1
    omiga = np.arange(0, dcache.means_exp.size, 1, np.int)
    _pull_arms(dcache, omiga)
    while omiga.size > 1:
        #  print dcache
        omiga = _ae_delete_certian_small_arms(dcache, omiga, k)
        k += 1
        _pull_arms(dcache, omiga)
    print dcache
    print 'omiga: ', omiga
    return


def successive_elimination(num_arms, pull=None, nu=_nu):
    """Successive elimination algorithm

    Args:
        num_arms: Number of arms
        pull: Function hook for generating datas(pull(arm_id))
        nu: The confidence of the algorithm
    """
    dcache = AlgCache(num_arms, pull)
    omiga = np.arange(0, dcache.n, 1, np.int)
    t = 1
    while omiga.size > 1:
        omiga = dcache.successive_elimination_pull(t, omiga, nu)
        t += 1
        if t % 1000 == 0:
            print 't = %d threshold = %f' % (
                t, _compute_successive_elimination_threshold(t, num_arms))
            print dcache.means
    print omiga
    return


def exponential_gap_elimination(num_arms, pull=None, nu=_nu):
    """Exponeential gap elimination algorithm."""
    dcache = AlgCache(num_arms, pull)
    omiga = np.arange(0, dcache.n, 1, np.int)
    r = 1
    epsilon = 1/8.0
    while omiga.size > 1:
        delta = nu / (50 * r * r * r)
        # print 'epsilon = %f delta = %f' % (epsilon, delta)
        omiga = dcache.exponential_gap_elimination_pull(epsilon, delta, omiga)
        epsilon /= 2.0
        r += 1
    print omiga
    return


def ucb(means=_means, vriances=_variances, delta=_delta, epsilon=_epsilon):
    """Upper confidence bound algorithm"""
# data = NormalData(means, vars)
    pass


def lucb(means=_means, variances=_variances, delta=_delta, epsilon=_epsilon):
    """LIL Upper confidence bound algorithm"""
    pass


def test():
    nd = NormalData(_means, _variances)
    print nd.means
    successive_elimination(nd.n, nd.pull)
    # exponential_gap_elimination(nd.n, nd.pull)
    # print nd.data[:nd.data_max]
    gap = np.amax(nd.means) - nd.means
    cnt = 0
    for el in gap:
        if not _is_zero(el):
            cnt += np.log(nd.n / _nu / el) / (el * el)
    print cnt
    print nd.data_max
    print nd.data_max / cnt
    return
