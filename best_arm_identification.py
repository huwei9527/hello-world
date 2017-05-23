import numpy as np
import data_structure as ds


def main():
    """Main"""
    pass


if __name__ == '__main__':
    main()

_float_type = ds._float_type


def max2(arr):
    max = arr[0]
    smax = arr[1]
    if (max < smax):
        max, smax = smax, max
    for i in range(2, arr.size, 1):
        el = arr[i]
        if (el > max):
            smax = max
            max = el
        elif (el > smax):
            smax = el
    return [max, smax]


class Configuration(object):
    """Documentation for Configuration

    """
    def __init__(
            self, size=100, confidence=0.1, epsilon=0.1, beta=1.0, sigma=0.25):
        super(Configuration, self).__init__()
        self.n = size
        self.v = confidence
        self.e = epsilon
        self.b = beta
        self.s = sigma
        return None

    def __str__(self):
        return ("n = %d v = %f e = %f b = %f s = %f" %
                (self.n, self.v, self.e, self.b, self.s))


class Arms(object):
    """Documentation for Arms

    """
    def __init__(self, size, pull):
        super(Arms, self).__init__()
        self.t = np.zeros(size, np.int)
        self.means = np.zeros(size, _float_type)
        self.arms = np.arange(0, size, 1, np.int)
        self.path = ds.IndexList()

        self._pull = pull
        return None

    def __getitem__(self, index):
        return self.arms[index]

    def __getattr__(self, name):
        if (name == 'size'):
            return self.arms.size
        else:
            raise AttributeError(name)
        return None

    def init(self, arms):
        self.arms = arms.arms.copy()
        # self.means = np.zeros(arms.means.size, _float_type)
        # self.t = np.zeros(self.means.size, _float_type)
        self.means = arms.means
        self.t = arms.t
        self._pull = arms._pull
        return None

    def set_arms_size(self, size):
        self.arms = np.zeros(size, np.int)
        return None

    def init_for_ls(self, size):
        self.means = ds.SortedList()
        for i in range(0, size, 1):
            self.t[i] = 1
            self.means.append(self._pull(i, 1))
        return None

    def delete(self, compare):
        i = 0
        flag = np.ones(self.arms.size, np.bool)
        del_flag = False
        while i < self.arms.size:
            if compare(self, self.arms[i]):
                flag[i] = False
                del_flag = True
            i += 1
        if del_flag:
            self.arms = self.arms[flag]
        return None

    def pull(self, arm_id, t):
        self.t[arm_id] += t
        self.means[arm_id] = self._pull(arm_id, t)
        return None

    def pull_all(self, t):
        for arm_id in self.arms:
            self.pull(arm_id, t)
        return None

    def pull_and_store(self, arm_id):
        self.t[arm_id] += 1
        self.means[arm_id] = self._pull(arm_id, 1)
        self.path.append(arm_id, self.means[arm_id])
        return None

    def pull_all_and_store(self):
        for arm_id in self.arms:
            self.pull_and_store(arm_id)
        return None

    def max_arms(self):
        max_id = self.arms[0]
        for arm_id in self.arms:
            if self.means[arm_id] > self.means[max_id]:
                max_id = arm_id
        return max_id

    def max_arms_mean(self):
        max_mean = self.means[self.arms[0]]
        for el in self.arms:
            if (self.means[el] > max_mean):
                max_mean = self.means[el]
        return max_mean

    def arms_median(self):
        return np.median(self.means[self.arms])


class LilStopCondition(object):
    """Documentation for LilStopCondition

    """
    def __init__(self, conf, arms):
        super(LilStopCondition, self).__init__()
        self._conf = conf
        self._arms = arms

        self._e_1 = 1 + self._conf.e
        self._d = (
            np.log(self._e_1) * np.power(
                self._conf.v * self._conf.e
                / (2 + self._conf.e), 1 / self._e_1))

        self._c1 = 1 + np.sqrt(self._conf.e)
        self._c2 = 2 * self._conf.s * self._conf.s * self._e_1
        self._c3 = 2 * self._conf.n / self._d

        self.lcb_of_max_arms = 0.0
        self.max_rest_ucb = 0.0
        return None

    def __str__(self):
        return ('%f %f' % (self.lcb_of_max_arms, self.max_rest_ucb))

    def _compute_b(self, t):
        return (self._c1 * np.sqrt(
            self._c2 * np.log(self._c3 * np.log(self._e_1 * t + 2)) / t))

    def is_stop(self):
        return (self.lcb_of_max_arms >= self.max_rest_ucb)


class LSBatch(LilStopCondition):
    """Documentation for LSBatch

    """
    def __init__(self, conf, arms):
        super(LSBatch, self).__init__(conf, arms)
        return None

    def next(self):
        if (self._arms.size == 1):
            self.lcb_of_max_arms = 100.0
            self.max_rest_ucb = -100.0
            return None
        b = self._compute_b(self._arms.t[self._arms.arms[0]])
        [max, smax] = max2(self._arms.means[self._arms.arms])
        self.lcb_of_max_arms = max - b
        self.max_rest_ucb = smax + b
        return None

    def pull_all(self, t=1):
        self._arms.pull_all(t)
        b = self._compute_b(self._arms.t[self._arms.arms[0]])
        i = 0
        size = self._arms.arms.size
        # Find the second largest element can be faster
        [max, smax] = max2(self._arms.means)
        self.lcb_of_max_arms = max - b
        self.max_rest_ucb = smax + b
        return None
        max = self._arms.means[self._arms.arms[0]]
        smax = self._arms.means[self._arms.arms[1]]
        if (max < smax):
            max, smax = smax, max
        for i in range(2, size):
            el = self._arms.means[self._arms.arms[i]]
            if (el > max):
                smax = max
                max = el
            elif (el > smax):
                smax = el
        self.lcb_of_max_arms = max - b
        self.max_rest_ucb = smax + b
        return None


class LSSerial(LilStopCondition):
    """Documentation for LSSerial

    """
    def __init__(self, conf, arms):
        super(LSSerial, self).__init__(conf, arms)
        self._b = ds.CachedList(self._compute_b)
        self._b.start_from_index_one()

        self._ucb = ds.SortedList()

        return None

    def _raw_pull(self, arm_id):
        self._arms.pull(arm_id, 1)
        self._ucb[arm_id] = (
            self._arms.means[arm_id] + self._b[self._arms.t[arm_id]])
        return None

    def _compute_ucb(self, arm_id):
        self._ucb[arm_id] = (
            self._arms.means[arm_id] + self._b[self._arms.t[arm_id]])
        return None

    def _compute_next_arms(self):
        self.max_mean_id = self._arms.means.argmax()
        self.max_rest_ucb_id = self._ucb.argmax_except_arms(self.max_mean_id)
        self.lcb_of_max_arms = (
            self._arms.means[self.max_mean_id]
            - self._b[self._arms.t[self.max_mean_id]])
        self.max_rest_ucb = self._ucb[self.max_rest_ucb_id]
        return None

    def init(self):
        self._arms.init_for_ls(self._conf.n)
        for i in range(0, self._conf.n, 1):
            self._ucb.append(self._arms.means[i] + self._b[1])
        self._compute_next_arms()
        return None

    def next(self):
        self._compute_ucb(self.max_mean_id)
        self._compute_ucb(self.max_rest_ucb_id)
        self._compute_next_arms()
        return None


class AlgorithmFrame(object):
    """Documentation for AlgorithmFrame

    """
    def __init__(self):
        super(AlgorithmFrame, self).__init__()
        self._arms = None
        self._conf = None
        self._pull = None

        self._basic_is_stop = None
        self._basic_run_str = None
        self._alg_init = None
        self._alg_next = None

        self.print_flag = True
        self.print_step = 1000
        self.alg_name = 'ALGFrame'
        self.str_alg = None
        return None

    def _basic_init(self, arms=None):
        self._arms = Arms(self._conf.n, self._pull)
        if (arms is not None):
            self._arms.init(arms)
        self._r = 1
        self.str_alg = self.alg_name
        self._basic_is_stop = self._is_stop
        return self._alg_init()

    def _basic_next(self):
        self._alg_next()
        self._r += 1
        return None

    def _result(self):
        return (self._arms.means.argmax())

    def _run(self, arms=None):
        if (self._basic_init(arms) < 0):
            return None
        while True:
            if ((self.print_flag) and (self._r % self.print_step == 0)):
                print ('%s[%d] %s' % (
                    self.str_alg, self._r, self._basic_run_str()))
            self._arms.pull_all(self._t)
            self._basic_next()
            if (self._basic_is_stop()):
                break
        return None

    def _run_steps(self, num_pulls=10000, arms=None):
        init_pulls = self._basic_init(arms)
        if (init_pulls < 0):
            print 'STOP in INIT Step'
            return None
        else:
            self._step_t = num_pulls - init_pulls
            self._step_r = 1
            self.num_pulls = init_pulls
        while True:
            if (self._step_t <= 0):
                break
            if (self.print_flag) and (self._step_r % 100 == 0):
                print '%s[%d %d]' % (self.str_alg, self._step_r, self._step_t)
            # self._arms.pull_all(1)
            self._arms.pull_all_and_store()
            self.num_pulls += self._arms.size
            self._t -= 1
            self._step_t -= self._arms.size
            self._step_r += 1
            if (self._t <= 0):
                self._basic_next()
                if (self._basic_is_stop()):
                    break
        # print 'size: %d' % self._arms.path.size
        return None


class Algorithm(AlgorithmFrame):
    """Documentation for Algorithm

    """
    def __init__(self, conf, pull):
        super(Algorithm, self).__init__()
        self._conf = conf
        self._pull = pull
        self._step_t = None

        self.alg_name = 'ALG'
        self.print_flag = True
        self.print_step = 1000
        return None

    def _init(self):
        return 0

    def _next(self):
        return None

    def _run_str(self):
        return ('')

    def _is_stop(self):
        return (self._arms.size == 1)

    def _ls_init(self):
        return None

    def _ls_next(self):
        return None

    def _is_ls_stop(self):
        return (self._ls.is_stop())

    def _alg_ls_init(self):
        self._conf.v /= 2.0
        ret = self._init()
        self.str_alg = self.alg_name + '(LS)'
        self._basic_is_stop = self._alg_is_ls_stop
        self._ls_init()
        return ret

    def _alg_ls_next(self):
        self._next()
        self._ls.next()
        self._ls_next()
        return None

    def _alg_ls_run_str(self):
        return ('%s %f %f' % (
            self._run_str(), self._ls.lcb_of_max_arms, self._ls.max_rest_ucb))

    def _alg_is_ls_stop(self):
        if (self._is_stop()):
            return True
        elif (self._is_ls_stop()):
            return True
        return False

    def run(self, t=1, num_pulls=None, arms=None):
        if (t == 1):
            self._alg_init = self._init
            self._alg_next = self._next
            self._basic_run_str = self._run_str
        else:
            self._alg_init = self._alg_ls_init
            self._alg_next = self._alg_ls_next
            self._basic_run_str = self._alg_ls_run_str
        if (self.print_flag):
            print 'Start ALG(%s)...' % (self.alg_name)
        if (num_pulls is None):
            self._run(arms)
        else:
            self._run_steps(num_pulls, arms)
        if (self.print_flag):
            print 'End ALG(%s).' % (self.alg_name)
        return self._result()

    def path(self):
        means = np.zeros(self._conf.n, _float_type)
        ret = np.zeros(self._arms.path.size, np.int)
        for i in range(0, self._arms.path.size, 1):
            el = self._arms.path[i]
            means[el[0]] = el[1]
            ret[i] = means.argmax()
        return ret

    def compute_time(self, means):
        pass


class Naive(Algorithm):
    """Documentation for Naive.

    """
    def __init__(self, conf, pull):
        super(Naive, self).__init__(conf, pull)
        self.alg_name = 'Naive'
        return

    def _compute_time(self):
        return np.int(np.ceil(
            4.0 * np.log(2 * self._conf.n / self._conf.v)
            / (self._conf.e * self._conf.e)))

    def _init(self, arms=None):
        self._t = self._compute_time()
        self.print_step = 1
        return 0

    def _is_stop(self):
        if (self._t > 1):
            return True
        else:
            return False

    def _run_str(self):
        return ('%d' % (self._t))

    def _ls_init(self):
        self._ls = LSBatch(self._conf, self._arms)
        self._t = 1
        self.print_step = 100
        return None

    def _ls_next(self):
        self._t = 1
        return None


class SuccessiveElimination(Algorithm):
    """Documentation for SuccessiveElimination.

    """
    def __init__(self, conf, pull):
        super(SuccessiveElimination, self).__init__(conf, pull)
        self._c = 4

        self._c1 = self._c * self._conf.n / self._conf.v

        self.alg_name = 'SE'
        return None

    def _compute_threshold(self):
        self._thd = 2 * np.sqrt(np.log(self._c1 * self._r * self._r) / self._r)
        return self._thd

    def _compare(self, arms, arm_id):
        return (self._max_mean - arms.means[arm_id] > self._thd)

    def _init(self):
        self._t = 1
        return 0

    def _next(self):
        self._max_mean = self._arms.max_arms_mean()
        self._compute_threshold()
        self._arms.delete(self._compare)
        self._t = 1
        return None

    def _run_str(self):
        return ('%d %f' % (self._arms.size, self._thd))

    def _ls_init(self):
        self._ls = LSBatch(self._conf, self._arms)
        self.print_step = 100
        return None

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


class MedianElimination(Algorithm):
    """Documentation for MedianElimination.

    """
    def __init__(self, conf, pull):
        super(MedianElimination, self).__init__(conf, pull)
        self.alg_name = 'ME'
        return None

    def _compute_time(self):
        return np.int(np.ceil(np.log(3.0 / self.d) / (self.e * self.e / 4.0)))

    def _compare(self, arms, arm_id):
        return (arms.means[arm_id] < self.median)

    def _delete_arms(self):
        self.median = self._arms.arms_median()
        self._arms.delete(self._compare)
        return None

    def _init(self, arms=None):
        self.e = self._conf.e / 4.0
        self.d = self._conf.v / 2.0
        self._t = self._compute_time()
        self.print_step = 1
        return 0

    def _next(self):
        self._delete_arms()
        self.e *= 0.75
        self.d /= 2.0
        self._t = self._compute_time()
        return None

    def _run_str(self):
        return ('%d %d %f %f' % (self._t, self._arms.size, self.e, self.d))

    def _ls_init(self):
        self._ls = LSBatch(self._conf, self._arms)
        return None

    def compute_time(self, means=None):
        ret = (
            self.conf.n * np.log(1 / self.conf.v)
            / (self.conf.e * self.conf.e))
        return ret


class ExponentialGapElimination(Algorithm):
    """Documentation for ExponentialGapElimination.

    """
    def __init__(self, conf, pull):
        super(ExponentialGapElimination, self).__init__(conf, pull)
        self.alg_name = 'EGE'
        return

    def _compute_time(self):
        return np.int(np.ceil(
            (2.0 / (self.e * self.e)) * np.log(2.0 / self.d)))

    def _compare(self, arms, arm_id):
        return arms.means[arm_id] < self.ref_mean

    def _delete_arms(self):
        ref_id = self._median_elimination(self.e / 2.0, self.d)
        self.ref_mean = self._arms.means[ref_id] - self.e
        self._arms.delete(self._compare)
        self.e /= 2.0
        return

    def _median_elimination(self, epsilon, delta):
        conf = Configuration(self._conf.n, delta, epsilon, 0.1, 0.25)
        me = MedianElimination(conf, self._pull)
        if (self._step_t is None):
            ret = me.run(0, arms=self._arms)
        else:
            ret = me.run(0, num_pulls=self._step_t, arms=self._arms)
            self._step_t -= me.num_pulls
        # self._arms.pull_all(1)
        # self._ls.next()
        return ret

    def _init(self, arms=None):
        print self._conf.n
        self.e = 1 / 8.0
        self.d = self._conf.v / (50.0 * self._r * self._r * self._r)
        self._t = self._compute_time()
        self.print_step = 1
        return 0

    def _next(self):
        self._delete_arms()
        self.e /= 2.0
        self.d = self._conf.v / (50.0 * self._r * self._r * self._r)
        self._t = self._compute_time()
        return False

    def _run_str(self):
        return ('%d %d %f %f' % (self._t, self._arms.size, self.e, self.d))

    def _ls_init(self):
        self._ls = LSBatch(self._conf, self._arms)
        return None

    def run_ls(self):
        print 'Start ExponentialGapElimination(LS)...'
        self._conf.v /= 2.0
        self._arms = Arms(self._conf.n, self._pull)
        ls = LSBatch(self._conf, self._arms)
        self._r = 1
        self.e = 1 / 8.0
        while True:
            self.d = self._conf.v / (50.0 * self._r * self._r * self._r)
            t = np.int(np.ceil(
                (2.0 / (self.e * self.e)) * np.log(2.0 / self.d)))
            print '(r = %d %d %f)' % (self._r, t, self.d)
            ls.pull_all(t)
            if (ls.is_stop()):
                break
            self._delete_arms()
            self.e /= 2.0
            self._r += 1
        print 'End ExponentialGapElimination(LS).'
        return self._arms.means.argmax()

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


class UCBStopCondition(object):
    """Documentation for LilBound

    """
    def __init__(self, conf, arms):
        super(UCBStopCondition, self).__init__()
        self._conf = conf
        self._arms = arms
        self._b = ds.CachedList(self._compute_b)
        self._b.start_from_index_one()
        self._ucb = ds.SortedList()
        self._max_t = 0
        self._sum_t = 0

        self._e_1 = self._conf.e + 1
        self._c = (
            (2 + self._conf.e)
            * np.power((1 / np.log(self._e_1)), self._e_1)
            / self._conf.e)
        self._lamda = np.power((2 + self._conf.b) / self._conf.b, 2)
        self._delta = (
            np.power(np.sqrt(self._conf.v + 0.25) - 0.5, 2) / self._c)

        self._c1 = (1 + self._conf.b) * (1 + np.sqrt(self._conf.e))
        self._c2 = 2 * np.power(self._conf.s, 2) * self._e_1
        return None

    def _compute_b(self, t):
        return (self._c1 *
                np.sqrt(self._c2 *
                        np.log(np.log(self._e_1 * t) / self._delta) / t))

    def _compute_ucb(self, arm_id):
        self._ucb[arm_id] = (
            self._arms.means[arm_id] + self._b[self._arms.t[arm_id]])
        return None

    def is_stop(self):
        return (self._max_t >= 1 + self._lamda * (self._sum_t - self._max_t))

    def init(self):
        for i in range(0, self._conf.n, 1):
            self._arms.pull(i, 1)
            self._sum_t += 1
            self._ucb.append(self._arms.means[i] + self._b[1])
        self._max_t = 1
        self.max_ucb_id = self._ucb.argmax()
        return None

    def next(self):
        arm_id = self._arms.arms[0]
        self._compute_ucb(arm_id)
        self._sum_t += 1
        if (self._arms.t[arm_id] > self._max_t):
            self._max_t = self._arms.t[arm_id]
        self.max_ucb_id = self._ucb.argmax()
        return None

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


class UpperConfidenceBound(Algorithm):
    """Documentation for UpperConfidenceBound.
    """
    def __init__(self, conf, pull):
        super(UpperConfidenceBound, self).__init__(conf, pull)
        self.alg_name = 'UCB'
        return None

    def _set_next_arms(self):
        self._arms.arms[0] = self._us.max_ucb_id
        return None

    def _init(self):
        self._us = UCBStopCondition(self._conf, self._arms)
        self._us.init()
        self._arms.set_arms_size(1)
        self._set_next_arms()
        self._t = 1
        if (self._us.is_stop()):
            print 'Stop in INIT STEP'
            return -1
        else:
            return self._arms.means.size
        return None

    def _is_stop(self):
        return (self._us.is_stop())

    def _next(self):
        self._us.next()
        self._set_next_arms()
        self._t = 1
        return None

    def _run_str(self):
        return ('%f' % (self._us._b[self._us._b.size - 1]))

    def _ls_init(self):
        self._ls = LSSerial(self._conf, self._arms)
        self._ls.init()
        return None

    def compute_time(self, means):
        return None


class LUCB(Algorithm):
    """Documentation for LUCB

    """
    def __init__(self, conf, pull):
        super(LUCB, self).__init__(conf, pull)
        self.alg_name = 'LUCB'
        return

    def _set_next_arms(self):
        self._arms.arms[0] = self._ls.max_mean_id
        self._arms.arms[1] = self._ls.max_rest_ucb_id
        return None

    def _init(self):
        self._ls = LSSerial(self._conf, self._arms)
        self._ls.init()
        self._arms.set_arms_size(2)
        self._set_next_arms()
        self._t = 1
        if (self._ls.is_stop()):
            print 'Stop in INIT STEP'
        else:
            return self._arms.means.size

    def _next(self):
        self._ls.next()
        self._set_next_arms()
        self._t = 1
        return None

    def _run_str(self):
        return ('%f %f' % (self._ls.lcb_of_max_arms, self._ls.max_rest_ucb))

    def _is_stop(self):
        return (self._ls.is_stop())
