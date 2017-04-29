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
        return

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
        self.sums = np.zeros(self.n, _float_type)
        self.means = np.zeros(self.n, _float_type)

    def __getitem__(self, index):
        return self.active_arms[index]

    def is_unique(self):
        return self.active_arms.size == 1

    def set_active_arms(self, arms):
        self.active_arms = arms.active_arms.copy()
        return

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

    def pull(self, arm_id):
        self.sums[arm_id] += self._pull(arm_id)
        self.t[arm_id] += 1
        self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
        return

    def pull_all(self, t=1):
        while t > 0:
            for arm_id in self.active_arms:
                self.sums[arm_id] += self._pull(arm_id)
                self.t[arm_id] += 1
            t -= 1
        for arm_id in self.active_arms:
            self.means[arm_id] = self.sums[arm_id] / self.t[arm_id]
        return

    def init_for_lil_stop_condition(self):
        self.means = ds.SortedList()
        for i in range(0, self.n, 1):
            self.sums[i] = self._pull(i)
            self.t[i] = 1
            self.means.append(self.sums[i] / self.t[i])
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


class FastArms(object):
    """Documentation for FastArms

    """
    def __init__(self):
        super(FastArms, self).__init__()
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
    def __init__(self, conf, pull):
        super(LSBatch, self).__init__(conf, pull)
        return

    def pull_all(self, t=1):
        self._arms.pull_all(t)
        b = self._compute_b(self._arms.t[self._arms.active_arms[0]])
        i = 0
        size = self._arms.active_arms.size
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
        return


class LSSerial(object):
    """Documentation for LSS

    """
    def __init__(self):
        super(LSSerial, self).__init__()
        return

    def pull(self):
        return


class Algorithm(object):
    """Algorithm cache, holding temporary data

    """
    def __init__(self, conf, pull):
        super(Algorithm, self).__init__()
        self.conf = conf
        self.pull = pull

    def compute_tie(self, means):
        """Compute the order of time in theory."""
        pass


class Naive(Algorithm):
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
        ls = LSBatch(self.conf, arms)
        self.r = 1
        while True:
            if self.r % 1000 == 0:
                print '(r = %d)(ls)' % (self.r)
            ls.pull_all()
            if (ls.stop_condition()):
                break
            self.r += 1
        return arms.means.argmax()


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

    def run(self):
        """Run the successive algorithm.
        """
        arms = Arms(self.conf.n, self.pull)
        self.r = 1
        while True:
            if (arms.is_unique()):
                break
            if (self.r % 10000 == 0):
                print '(r = %d)' % (self.r)
            arms.pull_all(1)
            self._delete_arms(arms)
            self.r += 1
        return arms.active_unique_arm_id()

    def run_ls(self):
        self.conf.v = self.conf.v / 2
        arms = Arms(self.conf.n, self.pull)
        ls = LSBatch(self.conf, arms)
        self.r = 1
        while True:
            if (self.r % 1000 == 0):
                print '(r = %d)(ls)' % (self.r)
            ls.pull_all(1)
            self._delete_arms(arms)
            if (ls.stop_condition()):
                break
            self.r += 1
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
        arms = Arms(self.conf.n, self.pull)
        if (eg_arms is not None):
            arms.set_active_arms(eg_arms)
        self.r = 1
        epsilon = self.conf.e / 4.0
        delta = self.conf.v / 2.0
        while True:
            if (arms.is_unique()):
                break
            t = np.log(3.0 / delta) / (epsilon * epsilon / 4.0)
            print '(t = %d)(me)' % t
            arms.pull_all(t)
            self._delete_arms(arms)
            epsilon *= 0.75
            delta /= 2.0
            self.r += 1
        return arms.active_unique_arm_id()

    def run_ls(self, eg_arms=None):
        self.conf.v = self.conf.v / 2
        arms = Arms(self.conf.n, self.pull)
        if (eg_arms is not None):
            arms.set_active_arms(eg_arms)
        ls = LSBatch(self.conf, arms)
        self.r = 1
        epsilon = self.conf.e / 4.0
        delta = self.conf.v / 2.0
        while True:
            t = np.log(3.0 / delta) / (epsilon * epsilon / 4.0)
            print '(t = %d)(me_ls)' % t
            ls.pull_all(t)
            if (ls.stop_condition()):
                break
            self._delete_arms(arms)
            epsilon *= 0.75
            delta /= 2.0
            self.r += 1
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
        arms = Arms(self.conf.n, self.pull)
        self.r = 1
        epsilon = 1 / 8.0
        while True:
            if (arms.is_unique()):
                break
            delta = self.conf.v / (50.0 * self.r * self.r * self.r)
            t = np.ceil((2.0 / (epsilon * epsilon)) * np.log(2.0 / delta))
            print '(r = %d, t = %d)(eg)' % (self.r, t)
            arms.pull_all(t)
            self._delete_arms(arms, epsilon, delta)
            epsilon /= 2.0
            self.r += 1
        return arms.active_arms[0]

    def run_ls(self):
        arms = Arms(self.conf.n, self.pull)
        ls = LSBatch(self.conf, arms)
        self.r = 1
        epsilon = 1 / 8.0
        while True:
            delta = self.conf.v / (50.0 * self.r * self.r * self.r)
            t = np.ceil((2.0 / (epsilon * epsilon)) * np.log(2.0 / delta))
            print '(r = %d, t = %d)(eg_ls)' % (self.r, t)
            ls.pull_all(t)
            if (ls.stop_condition()):
                break
            self._delete_arms(arms, epsilon, delta)
            epsilon /= 2.0
            self.r += 1
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


class LilBound(object):
    """Documentation for LilBound

    """
    def __init__(self, epsilon, confidence, beta=1.66, sigma=0.25):
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
    def __init__(self, conf):
        super(UCB, self).__init__()
        self.conf = conf
        self._ucb = ds.SortedList()
        return

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


class UpperConfidenceBound(Algorithm):
    """Upper confidence bound algorithm.
    """
    def __init__(self, conf, pull):
        super(UpperConfidenceBound, self).__init__(conf, pull)
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


class LUCB(Algorithm):
    """Documentation for LUCB

    """
    def __init__(self, conf, pull):
        super(LUCB, self).__init__(conf, pull)
        return

    def run(self):
        return


def test():
    return
