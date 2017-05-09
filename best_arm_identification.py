import numpy as np
import data_structure as ds


def main():
    """Main"""
    pass


if __name__ == '__main__':
    main()

_float_type = ds._float_type


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
        self.n = size
        self.active_arms = np.arange(0, self.n, 1, np.int)
        self._pull = pull
        self.t = np.zeros(self.n, np.int)
        self.means = np.zeros(self.n, _float_type)
        return None

    def __getitem__(self, index):
        return self.active_arms[index]

    def is_unique(self):
        return self.active_arms.size == 1

    def set_active_arms(self, arms):
        self.active_arms = arms.active_arms.copy()
        return None

    def delete(self, compare):
        i = 0
        flag = np.ones(self.active_arms.size, np.bool)
        del_flag = False
        while i < self.active_arms.size:
            if compare(self, self.active_arms[i]):
                flag[i] = False
                del_flag = True
            i += 1
        if del_flag:
            self.active_arms = self.active_arms[flag]
        return None

    def pull(self, arm_id, t):
        self.t[arm_id] += t
        self.means[arm_id] = self._pull(arm_id, t)
        return

    def pull_all(self, t):
        for arm_id in self.active_arms:
            self.pull(arm_id, t)
        return

    def init_for_lil_stop_condition(self):
        self.means = ds.SortedList()
        for i in range(0, self.n, 1):
            self.t[i] = 1
            self.means.append(self._pull(i, 1))
        return

    def arm_id_of_max_active_mean(self):
        max_id = self.active_arms[0]
        for arm_id in self.active_arms:
            if self.means[arm_id] > self.means[max_id]:
                max_id = arm_id
        return max_id

    def active_means_median(self):
        return np.median(self.means[self.active_arms])

    def active_unique_arm_id(self):
        return self.active_arms[0]

    def init_for_experiment(self):
        return None


class LilStopCondition(object):
    """Documentation for LilStopCondition

    """
    def __init__(self, conf, arms):
        super(LilStopCondition, self).__init__()
        self.conf = conf
        self._arms = arms

        self.e_1 = 1 + self.conf.e
        self._d = (
            np.log(self.e_1) * np.power(
                self.conf.v * self.conf.e / (2 + self.conf.e), 1 / self.e_1))

        self._c1 = 1 + np.sqrt(self.conf.e)
        self._c2 = 2 * self.conf.s * self.conf.s * self.e_1
        self._c3 = 2 * self.conf.n / self._d

        self.lcb_of_max_arms = 0.0
        self.max_rest_ucb = 0.0
        return

    def _compute_b(self, t):
        return (self._c1 * np.sqrt(
            self._c2 * np.log(self._c3 * np.log(self.e_1 * t + 2)) / t))

    def stop_condition(self):
        return (self.lcb_of_max_arms >= self.max_rest_ucb)


class LSBatch(LilStopCondition):
    """Documentation for LSBatch

    """
    def __init__(self, conf, arms):
        super(LSBatch, self).__init__(conf, arms)
        return None

    def pull_all(self, t=1):
        self._arms.pull_all(t)
        b = self._compute_b(self._arms.t[self._arms.active_arms[0]])
        i = 0
        size = self._arms.active_arms.size
        # Find the second largest element can be faster
        max = self._arms.means[self._arms.active_arms[0]]
        smax = self._arms.means[self._arms.active_arms[1]]
        if (max < smax):
            max, smax = smax, max
        for i in range(2, size):
            el = self._arms.means[self._arms.active_arms[i]]
            if (el > max):
                smax = max
                max = el
            elif (el > smax):
                smax = el
        self.lcb_of_max_arms = max - b
        self.max_rest_ucb = smax + b
        return None


class LSSerial(LilStopCondition):
    """Documentation for LSS

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

    def init(self):
        self._arms.init_for_lil_stop_condition()
        for i in range(0, self.conf.n, 1):
            self._ucb.append(self._arms.means[i] + self._b[1])
        max_mean_id = self._arms.means.argmax()
        max_rest_ucb_id = self._ucb.argmax_except_arms(max_mean_id)
        self.lcb_of_max_arms = (
            self._arms.means[max_mean_id] - self._b[self._arms.t[max_mean_id]])
        self.max_rest_ucb = self._ucb[max_rest_ucb_id]
        return ([max_mean_id, max_rest_ucb_id])

    def pull(self, max_mean_id=None, max_rest_ucb_id=None):
        if (max_mean_id is not None):
            self._raw_pull(max_mean_id)
        self._raw_pull(max_rest_ucb_id)
        max_mean_id = self._arms.means.argmax()
        max_rest_ucb_id = self._ucb.argmax_except_arms(max_mean_id)
        self.lcb_of_max_arms = (
            self._arms.means[max_mean_id] - self._b[self._arms.t[max_mean_id]])
        self.max_rest_ucb = self._ucb[max_rest_ucb_id]
        return [max_mean_id, max_rest_ucb_id]

    def init_for_ucb(self):
        self._us = UCBStopCondition(self.conf, self.pull)
        self.init()
        for i in range(0, self.conf.n, 1):
            self._us._ucb.append(self._arms.means[i] + self._us._b[1])
        return self._us._ucb.argmax()

    def pull_for_ucb(self, arm_id):
        self.pull(None, arm_id)
        self._us._ucb[arm_id] = (
            self._arms.means[arm_id] + self._us._b[self._arms.t[arm_id]])
        return self._us._ucb.argmax()


class Algorithm(object):
    """Algorithm cache, holding temporary data

    """
    def __init__(self, conf, pull):
        super(Algorithm, self).__init__()
        self.conf = conf
        self.pull = pull

    def compute_time(self, means):
        """Compute the order of time in theory."""
        pass


class Naive(Algorithm):
    """Naive algorithm.

    """
    def __init__(self, conf, pull):
        super(Naive, self).__init__(conf, pull)
        return

    def _compute_time(self):
        return np.int(np.ceil(
            4.0 * np.log(2 * self.conf.n / self.conf.v)
            / (self.conf.e * self.conf.e)))

    def _init(self, arms):
        self.r = self._compute_time()
        return None

    def _next(self, arms):
        if (self.r > 0):
            return True
        else:
            return False
        return None

    def run(self, num_pulls=None):
        print 'Start Naive ...'
        arms = Arms(self.conf.n, self.pull)
        self._init(arms)
        if (num_pulls is None):
            print 'Start to pull each arms(%d)' % (self.r)
            arms.pull_all(self.r)
        else:
            while True:
                if self._next():
                    break
                arms.pull_all(1)
                self.r -= 1
        print 'End Naive.'
        return arms.means.argmax()

    def run_ls(self):
        print 'Start Naive(LS)...'
        self.conf.v /= 2
        arms = Arms(self.conf.n, self.pull)
        ls = LSBatch(self.conf, arms)
        self.r = 1
        while True:
            if self.r % 1000 == 0:
                print '(r = %d %f %f %f)' % (
                    self.r,
                    ls.lcb_of_max_arms,
                    ls.max_rest_ucb,
                    ls._compute_b(self.r))
            ls.pull_all()
            if (ls.stop_condition()):
                break
            self.r += 1
        print 'End Naive(LS)'
        return arms.means.argmax()

    def run_n(self, n):
        arms = Arms(self.conf.n, self.pull)
        self.r = 0
        while True:
            if self.r > n:
                break
            arms.pull_all(1)
        return None

    def run_ls_n(self, n):
        return None


class SuccessiveElimination(Algorithm):
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
        """
        self.threshold = 2 * np.sqrt(
            np.log(self.c1 * self.r * self.r) / self.r)
        return

    def _compare(self, arms, arm_id):
        return (
            self.max_mean - arms.means[arm_id] > self.threshold)

    def _delete_arms(self, arms):
        ref_id = arms.arm_id_of_max_active_mean()
        self.max_mean = arms.means[ref_id]
        self.compute_threshold()
        return arms.delete(self._compare)

    def _next(self):
        return None

    def run(self):
        print 'Start SuccessiveElimination...'
        arms = Arms(self.conf.n, self.pull)
        self.r = 1
        while True:
            if (arms.is_unique()):
                break
            if (self.r % 1000 == 0):
                print '(r = %d %f %d)' % (
                    self.r, self.threshold, arms.active_arms.size)
            arms.pull_all(1)
            self._delete_arms(arms)
            self.r += 1
        print 'End SuccessiveElimination'
        return arms.active_unique_arm_id()

    def run_ls(self):
        print 'Start SuccessiveElimination(LS)...'
        self.conf.v = self.conf.v / 2
        arms = Arms(self.conf.n, self.pull)
        ls = LSBatch(self.conf, arms)
        self.r = 1
        while True:
            if (self.r % 1000 == 0):
                print '(r = %d %d %f %f %f)' % (
                    self.r,
                    arms.active_arms.size,
                    ls.lcb_of_max_arms,
                    ls.max_rest_ucb,
                    ls._compute_b(self.r))
            ls.pull_all(1)
            self._delete_arms(arms)
            if (ls.stop_condition()):
                break
            self.r += 1
        print 'End SuccessiveElimination(LS).'
        return arms.means.argmax()

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
    """Median elimination.

    """
    def __init__(self, conf, pull):
        super(MedianElimination, self).__init__(conf, pull)
        return

    def _compare(self, arms, arm_id):
        return (arms.means[arm_id] < self.median)

    def _delete_arms(self, arms):
        self.median = arms.active_means_median()
        arms.delete(self._compare)
        return

    def run(self, eg_arms=None):
        print 'Start MedianElimination...'
        arms = Arms(self.conf.n, self.pull)
        if (eg_arms is not None):
            arms.set_active_arms(eg_arms)
        self.r = 1
        epsilon = self.conf.e / 4.0
        delta = self.conf.v / 2.0
        while True:
            if (arms.is_unique()):
                break
            t = np.int(np.ceil(
                np.log(3.0 / delta) / (epsilon * epsilon / 4.0)))
            print '(t = %d %d %f %f)' % (
                t, arms.active_arms.size, epsilon, delta)
            arms.pull_all(t)
            self._delete_arms(arms)
            epsilon *= 0.75
            delta /= 2.0
            self.r += 1
        print 'End MedianElimination.'
        return arms.active_unique_arm_id()

    def run_ls(self, eg_arms=None):
        print 'Start MedianElimination(LS)...'
        self.conf.v = self.conf.v / 2
        arms = Arms(self.conf.n, self.pull)
        if (eg_arms is not None):
            arms.set_active_arms(eg_arms)
        ls = LSBatch(self.conf, arms)
        self.r = 1
        epsilon = self.conf.e / 4.0
        delta = self.conf.v / 2.0
        while True:
            t = np.int(np.ceil(
                np.log(3.0 / delta) / (epsilon * epsilon / 4.0)))
            print '(t = %d %d %f %f)' % (
                t, arms.active_arms.size, epsilon, delta)
            ls.pull_all(t)
            if (ls.stop_condition()):
                break
            self._delete_arms(arms)
            epsilon *= 0.75
            delta /= 2.0
            self.r += 1
        print 'End MedianElimination(LS).'
        return arms.means.argmax()

    def compute_time(self, means=None):
        ret = (
            self.conf.n * np.log(1 / self.conf.v)
            / (self.conf.e * self.conf.e))
        return ret


class ExponentialGapElimination(Algorithm):
    """Exponential gap elimination algorithm

    """
    def __init__(self, conf, pull):
        super(ExponentialGapElimination, self).__init__(conf, pull)
        return

    def _compare(self, arms, arm_id):
        return arms.means[arm_id] < self.ref_mean

    def _delete_arms(self, arms, epsilon, delta):
        ref_id = self._median_elimination(arms, epsilon / 2.0, delta)
        self.ref_mean = arms.means[ref_id] - epsilon
        arms.delete(self._compare)
        return

    def _median_elimination(self, arms, epsilon, delta):
        conf = Configuration(self.conf.n, delta, epsilon, 0.1, 0.25)
        me = MedianElimination(conf, self.pull)
        return me.run_ls(arms)

    def run(self):
        print 'Start ExponentialGapElimination...'
        arms = Arms(self.conf.n, self.pull)
        self.r = 1
        epsilon = 1 / 8.0
        while True:
            if (arms.is_unique()):
                break
            delta = self.conf.v / (50.0 * self.r * self.r * self.r)
            t = np.int(np.ceil(
                (2.0 / (epsilon * epsilon)) * np.log(2.0 / delta)))
            print '(r = %d %d %f)' % (self.r, t, delta)
            arms.pull_all(t)
            self._delete_arms(arms, epsilon, delta)
            epsilon /= 2.0
            self.r += 1
        print 'End ExponentialGapElimination.'
        return arms.active_arms[0]

    def run_ls(self):
        print 'Start ExponentialGapElimination(LS)...'
        self.conf.v /= 2
        arms = Arms(self.conf.n, self.pull)
        ls = LSBatch(self.conf, arms)
        self.r = 1
        epsilon = 1 / 8.0
        while True:
            delta = self.conf.v / (50.0 * self.r * self.r * self.r)
            t = np.int(np.ceil(
                (2.0 / (epsilon * epsilon)) * np.log(2.0 / delta)))
            print '(r = %d %d %f)' % (self.r, t, delta)
            ls.pull_all(t)
            if (ls.stop_condition()):
                break
            self._delete_arms(arms, epsilon, delta)
            epsilon /= 2.0
            self.r += 1
        print 'End ExponentialGapElimination(LS).'
        return arms.means.argmax()

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
        self.conf = conf
        self._arms = arms
        self._b = ds.CachedList(self._compute_b)
        self._b.start_from_index_one()
        self._ucb = ds.SortedList()
        self._max_t = 0
        self._sum_t = 0

        self.e_1 = self.conf.e + 1
        self._c = (
            (2 + self.conf.e)
            * np.power((1 / np.log(self.e_1)), self.e_1)
            / self.conf.e)
        self._lamda = np.power((2 + self.conf.b) / self.conf.b, 2)
        self._delta = (
            np.power(np.sqrt(self.conf.v + 0.25) - 0.5, 2) / self._c)

        self._c1 = (1 + self.conf.b) * (1 + np.sqrt(self.conf.e))
        self._c2 = 2 * np.power(self.conf.s, 2) * self.e_1
        return None

    def stop_condition(self):
        return (self._max_t >= 1 + self._lamda * (self._sum_t - self._max_t))

    def _compute_b(self, t):
        return (self._c1 *
                np.sqrt(self._c2 *
                        np.log(np.log(self.e_1 * t) / self._delta) / t))

    def init(self):
        for i in range(0, self.conf.n, 1):
            self._arms.pull(i, 1)
            self._sum_t += 1
            self._ucb.append(self._arms.means[i] + self._b[1])
        self._max_t = 1
        return self._ucb.argmax()

    def pull(self, arm_id):
        self._arms.pull(arm_id, 1)
        self._ucb[arm_id] = (
            self._arms.means[arm_id] + self._b[self._arms.t[arm_id]])
        self._sum_t += 1
        if (self._arms.t[arm_id] > self._max_t):
            self._max_t = self._arms.t[arm_id]
        return self._ucb.argmax()

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
    """Upper confidence bound algorithm.int 
    """
    def __init__(self, conf, pull):
        super(UpperConfidenceBound, self).__init__(conf, pull)
        return None

    def run(self):
        print 'Start UCB...'
        arms = Arms(self.conf.n, self.pull)
        us = UCBStopCondition(self.conf, arms)
        arm_id = us.init()
        self.r = 1
        while True:
            if us.stop_condition():
                break
            arm_id = us.pull(arm_id)
            self.r += 1
            if (self.r % 1000 == 0):
                print '(r = %d %f)' % (self.r, us._b[us._b.size - 1])
        print 'End UCB.'
        return arm_id

    def run_ls(self):
        print 'Start UCB(LS)...'
        self.conf.v /= 2
        arms = Arms(self.conf.n, self.pull)
        ls = LSSerial(self.conf, arms)
        arm_id = ls.init_for_ucb()
        self.r = 1
        while True:
            if ls.stop_condition():
                break
            arm_id = ls.pull_for_ucb(arm_id)
            self.r += 1
            if (self.r % 1000 == 0):
                print '(r = %d %f %f)' % (
                    self.r,
                    ls.lcb_of_max_arms,
                    ls.max_rest_ucb)
        print 'End UCB(LS).'
        return arm_id

    def compute_time(self, means):
        return None


class LUCB(Algorithm):
    """Documentation for LUCB

    """
    def __init__(self, conf, pull):
        super(LUCB, self).__init__(conf, pull)
        return

    def run(self):
        print 'Start LUCB...'
        arms = Arms(self.conf.n, self.pull)
        ls = LSSerial(self.conf, arms)
        self.r = 1
        [max_mean_id, max_rest_ucb_id] = ls.init()
        while True:
            if (ls.stop_condition()):
                break
            [max_mean_id, max_rest_ucb_id] = (
                ls.pull(max_mean_id, max_rest_ucb_id))
            self.r += 1
            if (self.r % 1000 == 0):
                print '(r = %d %f %f)' % (
                    self.r,
                    ls.lcb_of_max_arms,
                    ls.max_rest_ucb)
        print 'End LUCB'
        return arms.means.argmax()


def test():
    return
