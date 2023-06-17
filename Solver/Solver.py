from .IntervalTree import *

def movement(xs_last, u_last, xs, u):
    """Computes |u_last - u| / |u_last + u| with linear interpolation."""
    xs_combined = np.concatenate((xs_last, xs))
    u_last_i = np.interp(xs_combined, xs_last, u_last)
    u_i = np.interp(xs_combined, xs, u)
    return np.max(np.abs(u_last_i - u_i)) / (np.max(np.abs(u_last_i + u_i)) + 0.01)


class Solver:
    def __init__(self, l, r, p, q, f0, z_l0, z_l1, z_r0, z_r1, G_l, G_r):
        """
        Solves the equation
            u'' + p u' + q u = f0
        on [l, r] subject to the boundary condition
            z_l0 u(l) + z_l1 u'(l) = G_l
        and
            z_r0 u(r) + z_r1 u'(r) = G_r.
        """
        # First we get the inhomogeneous part,
        #  u_i = ax^2 + bx + c
        # This produces an underdetermined linear equation.
        M_i = np.array([
            [z_l0 * l**2 + z_l1 * 2*l, z_l0 * l + z_l1, z_l0],
            [z_r0 * r**2 + z_r1 * 2*r, z_r0 * r + z_r1, z_r0],
        ])
        (a, b, c) = np.linalg.lstsq(M_i, np.array([G_l, G_r]), rcond=None)[0]

        # Evaluate the relevant reductions
        self.u_i = lambda x: a*x*x + b*x + c
        self.du_i = lambda x: 2*a*x + b
        IntervalTree.f = lambda x: f0(x) - (2*a + p(x) * self.du_i(x) + q(x) * self.u_i(x))
        if abs(z_l0) >= abs(z_l1) or abs(z_r0) >= abs(z_r1):
            IntervalTree.g_l = lambda x: z_l0 * (x - l) - z_l1
            self.dg_l = lambda x: z_l0
            IntervalTree.g_r = lambda x: z_r0 * (x - r) - z_r1
            self.dg_r = lambda x: z_r0
            self.s = (z_r0 * r + z_r1) * z_l0 - (z_l0 * l + z_l1) * z_r0
            qq = q
        else:
            IntervalTree.g_l = lambda x: z_l1 * np.cosh(x - l) - z_l0 * np.sinh(x - l)
            self.dg_l = lambda x: z_l1 * np.sinh(x - l) - z_l0 * np.cosh(x - l)
            IntervalTree.g_r = lambda x: z_r1 * np.cosh(x - r) - z_r0 * np.sinh(x - r)
            self.dg_r = lambda x: z_r1 * np.sinh(x - r) - z_r0 * np.cosh(x - r)
            self.s = (z_l1 * z_r1 - z_l0 * z_r0) * np.sinh(l-r) + (z_l0 * z_r1 - z_l1 * z_r0) * np.cosh(l-r)
            qq = lambda x: q(x) - 1
        IntervalTree.psi_l = lambda x: (p(x) * self.dg_r(x) + qq(x) * IntervalTree.g_r(x))/self.s
        IntervalTree.psi_r = lambda x: (p(x) * self.dg_l(x) + qq(x) * IntervalTree.g_l(x))/self.s
        self.tree = Leaf(l,r)

    def integrate(self):
        """
        Solves the equation with the current mesh. Returns a tuple (x, u, u').
        """
        # Local solutions
        self.tree.upwards()
        self.tree.downwards_root()

        Jl = 0
        for leaf in self.tree.sweep_left():
            leaf.Jl = Jl
            Jl += leaf.delta_l + leaf.lmbda_l * leaf.alpha_l + leaf.lmbda_r * leaf.beta_l

        Jr = 0
        for leaf in self.tree.sweep_right():
            leaf.Jr = Jr
            Jr += leaf.delta_r + leaf.lmbda_l * leaf.alpha_r + leaf.lmbda_r * leaf.beta_r

        xs = np.array([], dtype=float)
        u = np.array([], dtype=float)
        du = np.array([], dtype=float)
        # Concatenate the solutions
        for leaf in self.tree.sweep_left():
            xs = np.concatenate((xs, leaf.tau))
            u = np.concatenate((u,
                self.u_i(leaf.tau) \
                    + leaf.g_r_eval/self.s * (leaf.Jl + Local.IL @ (leaf.g_l_eval * leaf.sigma) * (leaf.r - leaf.l) / 2)
                    + leaf.g_l_eval/self.s * (leaf.Jr + Local.IR @ (leaf.g_r_eval * leaf.sigma) * (leaf.r - leaf.l) / 2)))
            du = np.concatenate((du,
                self.du_i(leaf.tau) \
                    + self.dg_r(leaf.tau)/self.s * (leaf.Jl + Local.IL @ (leaf.g_l_eval * leaf.sigma) * (leaf.r - leaf.l) / 2)
                    + self.dg_l(leaf.tau)/self.s * (leaf.Jr + Local.IR @ (leaf.g_r_eval * leaf.sigma) * (leaf.r - leaf.l) / 2)))
        return xs, u, du

    def solve(self):
        xs, u = np.array([self.tree.l, self.tree.r]), np.array([0.,0.])
        final_doubled = False
        while True:
            # Compute solution
            xs_last, u_last = xs, u
            xs, u, du = self.integrate()

            # yield xs, u, du   # For debug use

            if final_doubled:
                # We're done. Note that this logic is different from
                # the paper which is a bit weird.
                break

            # If it changed by a lot, we refine
            if movement(xs_last, u_last, xs, u) > Local.TOL:
                S = max(leaf.monitor() for leaf in self.tree.sweep_left()) / 2**Local.C
                r = self.tree.refine(S)
                if r is not None:
                    self.tree = r
            elif not final_doubled:
                final_doubled = True
                # Do a final round that splits every subinterval
                self.tree.double()

        return xs, u, du
