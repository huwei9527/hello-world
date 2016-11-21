#!/usr/bin/python


def fib(n):
    """return a list, containing Fibonacci array from 1 to n"""
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result


def ask_ok(prompt, retries=4, complaint='Yes or no, please!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise IOError('resusenik user')
        print complaint


def make_incrementor(n):
    return lambda x: x + n


a = [2, 1, 3, 4]
l = [5, 6]
b = [[3*x, x**2] for x in a]
m = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]


def scope_test():
    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_global()
    print "After global assignment:", spam


class MyClass:
    """A simple example class"""

    i = 12345

    def f(self, num):
        self.i = num
        return "Hello world"

def average(values):
    """Computes the arithmetic mean of a list of numbers.
    
    >>> print average({20, 30, 70})
    40.0
    """
    
    return sum(values) / len(values)

import numpy
a = numpy.eye(4)
print a.shape
