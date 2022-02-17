import time

t1 = time.perf_counter()

from sympy import Symbol, Dummy, cse, parse_expr, diff
import sympy.parsing.sympy_parser as spp

def flint(eq):
    reps = {}
    e = eq.replace(
        lambda x: x.is_Float and x == int(x),
        lambda x: reps.setdefault(x, Dummy()))
    return e.xreplace({v: int(k) for k, v in reps.items()})


expr = 'sin(x)^3'
#diffexpr = parse_expr(expr, evaluate=False)
symbolis = ['x', 'y']
syms = {}

for sym in symbolis:
    syms[sym] = Symbol(sym)


symexpr = parse_expr(expr, local_dict=syms, transformations=(spp.convert_xor, spp.auto_number), evaluate=False)
#symexpr = flint(symexpr.simplify()).simplify()
symexpr = flint(symexpr.simplify()).simplify()

subexprs = cse(symexpr)

diffsymexpr = diff(symexpr, syms['x'])

t2 = time.perf_counter()

print(t2-t1)
print(symexpr)
print(subexprs)
print(diffsymexpr)
