import numpy as np
import matplotlib.pyplot as plt
import sys


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
_means = np.arange(0, 1.1, 0.2, dtype = _float_type)
#  _means = np.random.permutation(_means)
# Default variance of the normal random variables
_vars = np.ones(_means.size, dtype = _float_type) * 0.25
_num_pulls = 70


def _is_zero(float_num):
    """Decide whether the given float number is zero"""
    return (np.fabs(float_num) < precision)


def _is_equal(floata, floatb):
    """Decide whether two floats are equal accroding to the precision"""
    return _is_zero(floata, floatb)


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
    ep_plus_one = epsilon + 1;
    log_v = np.log10(ep_plus_one * t + 2)
    loglog_v = np.log10(log_v / delta)
    return (1 + np.sqrt(epsilon)) * np.sqrt(ep_plus_one * loglog_v / (2 * t))


class NormalData(object):
    """Normal data"""

    def __init__(self, means, vars, num_pulls=_num_pulls):
        """Init function"""
        self.means = means.copy()
        self.vars = vars.copy()
        self._mem_step = int(num_pulls * _compute_h_1(self.means))
        self.data = np.zeros(self._mem_step, dtype=_float_type)
        self.data_id = np.zeros(self.data.shape, dtype=np.int)
        self.data_max = 0
        return

    def __str__(self):
        return ((
            "means: %s\n"
            "variances: %s\n"
            "H1: %f\n"
            "(data_max/max):(%d/%d)\n") % (
                self.means, 
                self.vars, 
                _compute_h_1(self.means), 
                self.data_max, self.data.size))


    def pull(self, arm_id):
        """Pull the arm_id-th arm without checking the legality of the index.
        
        You need to make sure that arm_id is in the range.
        """
        assert arm_id < self.means.size
        if not (self.data_max < self.data.size):
            self.data.resize(2 * self.data.size)
            self.data_id.resize(self.data.size)
        arm_id = int(arm_id)
        # non-central normal distribution: var * n + mu
        self.data[self.data_max] = np.random.normal(self.means[arm_id],
                self.vars[arm_id])
        # maintain the index which arm the value belongs to
        self.data_id[self.data_max] = arm_id
        self.data_max+=1
        return self.data[arm_id]

    def save(self, fname='NormalData'):
        np.savez(fname, means=self.means, vars=self.vars,
                data=self.data[:self.data_max], id=self.data_id[:self.data_max])
        return

    def load(self, fname='NormalData.npz'):
        with np.load(fname) as d:
            self.means = d['means']
            self.vars = d['vars']
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
        plt.hist(b, bins=np.ceil(a[arm_id]/10.0))
        plt.show()


data = NormalData(_means, _vars)


class AlgCache:
    """Cache useful parameters used during the computing."""

    def __init__(self, num_arms, pull_hook=None, conf_hook=None):
        """Init function"""
        self.means_exp = np.zeros(num_arms, _float_type)
        self.conf = np.zeros(self.means_exp.size, _float_type)
        self.t = np.zeros(self.means_exp.size, np.int)
        self.h = 0
        self.l = 1
        self.pull_hook = pull_hook
        self.conf_hook = conf_hook
        return

    def __str__(self):
        """format printing"""
        return "measns_exp: %s\nconf: %s\nt: %s\n(h, l): (%d, %d)" % (self.means_exp,
                self.conf, self.t, self.h, self.l)

    def _find_max_experience_mean_id(self):
        return self.means_exp.argmax()

    def _find_max_confidence_id(self):
        old_conf = self.conf[self.h]
        self.conf[self.h] = self.conf.min() - 1.0
        ret_max_conf_id = self.conf.argmax()
        self.conf[self.h] = old_conf
        return ret_max_conf_id

    def pull(self, arm_id):
        """Pull the arm_id-th arm and maintain the parameters."""
        if self.pull_hook == None:
            return
        assert arm_id >= 0 and arm_id < self.means_exp.size
        out_val = self.pull_hook(arm_id)
        frac = self.t[arm_id] / (self.t[arm_id] + 1.0)
        old_mean = self.means_exp[arm_id]
        self.means_exp[arm_id] = (frac * old_mean + (1 - frac) * out_val)
        self.t[arm_id] += 1
        # compute h = arg max {\mu}
        # compute l = arg max_{i\h} {\mu + c}
        if self.conf_hook == None:
            return
        self.conf[arm_id] = (self.means_exp[arm_id] +
                self.conf_hook(self.t[arm_id]))
        self.h = self._find_max_experience_mean_id()
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


def _pull_arms(data_cache, omiga, r=1):
    """Pull all the arms r times in the set omiga"""
    while r > 0:
        for i in range(0, omiga.size, 1):
            data_cache.pull(omiga[i])
        r -= 1
    #  print data_cache
    return


class AEConfidence:
    """Compute the confidence of the action elimination"""
    def __init__(self, delta=_delta, epsilon=_epsilon):
        self.delta = _delta
        self.epsilon = _epsilon
        return

    def compute(self, t):
        return (2 * _compute_u(t, self.delta, self.epsilon))

    def compute_positive_rate(self):
        ep_plus_one = 1 + self.epsilon
        log_v = np.log10(ep_plus_one)
        tmp = np.power(1/log_v, ep_plus_one)
        tmp = 1 - ((2 + self.epsilon) * 2 * tmp * self.delta / self.epsilon)
        return tmp
        

def action_elimination(num_arms, pull=None, delta=_delta, epsilon=_epsilon):
    """Nonadaptive LS algorithm"""
    conf_hook = AEConfidence(delta/num_arms, epsilon)
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


def ucb(means=_means, vars=_vars, delta=_delta, epsilon=_epsilon):
    """Upper confidence bound algorithm"""
    data = NormalData(means, vars)
    pass


def lucb(means=_means, vars=_vars, delta=_delta, epsilon=_epsilon):
    """LIL Upper confidence bound algorithm"""
    pass
