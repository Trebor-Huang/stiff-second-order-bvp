import numpy as np
from .Solver import Solver, Local
import time

def Sum(x, xss, vs):
    return sum((np.interp(x, xs, v) for xs, v in zip(xss, vs)))

def newton(l, r, f, f2, f3, z_l0, z_l1, z_r0, z_r1, G_l=0, G_r=0):
    """
    Solves the equation on [l,r]
        u''(x) = f(x, u(x), u'(x))
    Given f and its derivatives at the second and third position.
    The boundary condition is
        zl0 * u(l) + zl1 * u'(l) = G_l,
        zr0 * u(r) + zr1 * u'(r) = G_r.
    """
    M_i = np.array([
        [z_l0 * l**2 + z_l1 * 2*l, z_l0 * l + z_l1, z_l0],
        [z_r0 * r**2 + z_r1 * 2*r, z_r0 * r + z_r1, z_r0],
    ])
    (a, b, c) = np.linalg.lstsq(M_i, np.array([G_l, G_r]), rcond=None)[0]

    # Evaluate the relevant reductions
    u_i = lambda x: a*x*x + b*x + c
    du_i = lambda x: 2*a*x + b

    xss = []
    vs = []
    dvs = []
    ddvs = []

    xs = np.array([l,r])
    v = np.array([0.,0.])
    dv = np.array([0.,0.])
    ddv = np.array([0.,0.])

    g = lambda x, y, dy: f(x, y+u_i(x), dy+du_i(x))
    g2 = lambda x, y, dy: f2(x, y+u_i(x), dy+du_i(x))
    g3 = lambda x, y, dy: f3(x, y+u_i(x), dy+du_i(x))

    # We solve the linear equation
    #  v'' - f3(x, u(x), u'(x)) v' - f2(x, u(x), u'(x)) v = u'' - f(x, u, u')
    while True:
        xs, v, dv = Solver(l, r,
            lambda x: -g3(x, Sum(x, xss, vs), Sum(x, xss, dvs)),
            lambda x: -g2(x, Sum(x, xss, vs), Sum(x, xss, dvs)),
            lambda x: Sum(x, xss, ddvs)+2*a - g(x, Sum(x, xss, vs), Sum(x, xss, dvs)),
            z_l0, z_l1, z_r0, z_r1, 0, 0).solve()
        ddv = Sum(xs, xss, ddvs)+2*a - g(xs, Sum(xs, xss, vs), Sum(xs, xss, dvs)) \
            + g3(xs, Sum(xs, xss, vs), Sum(xs, xss, dvs)) * dv \
            + g2(xs, Sum(xs, xss, vs), Sum(xs, xss, dvs)) * v

        if np.max(np.abs(v)) < Local.TOL:
            break

        xss.append(xs)
        vs.append(-v)
        dvs.append(-dv)
        ddvs.append(-ddv)

    if len(xss) == 0:
        # We don't have a mesh, and the initial quadratic guess is already
        # exact enough. So we leave the question for the user.
        # Alternatively we can just use some fixed mesh like np.linspace(l,r).
        raise Exception("Just use a quadratic function!")

    xs = np.unique(np.concatenate(xss))
    u = Sum(xs, xss, vs) + u_i(xs)
    du = Sum(xs, xss, dvs) + du_i(xs)
    ddu = Sum(xs, xss, ddvs) + 2*a
    return xs, u, du, ddu

