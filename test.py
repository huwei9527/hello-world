for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print n, 'equals', x, '*', n//x
            break
    else:
        print n, 'is a prime number'


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


scope_test()
print "In global scope:", spam


class MyClass:
    """A simple example class"""

    i = 12345

    def f(self, num):
        self.i = num
        return "Hello world"


c = MyClass()
d = MyClass()
c.f(100)
print c.i
print MyClass.i
c.attr1 = 10
c.attr2 = 20
print c.attr1
print c.attr2
