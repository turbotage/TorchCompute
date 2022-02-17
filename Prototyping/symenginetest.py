import time

t1 = time.perf_counter()

import symengine

from symengine import Symbol, Dummy, sympify, cse
from symengine import functions

def flint(eq):
    reps = {}
    e = eq.replace(
        lambda x: x.is_Float and x == int(x),
        lambda x: reps.setdefault(x, Dummy()))
    return e.xreplace({v: int(k) for k, v in reps.items()})

expr = '1.0*sin(x)^2+2*cos(x)^2+1.045e-6+I+exp(I)+tan(sin(x))-exp(tan(sin(x)))+6*tanh(exp(sin(x)))'

symped = flint(sympify(expr))

csed = cse([expr])

t2 = time.perf_counter()

print(t2-t1)

print(symped)
print(csed)
