import numpy as np

class Local:
    """Singleton class recording initialized information."""
    @staticmethod
    def initialize(K, C, TOL):
        ## Initialization of various constants
        Local.K = K
        Local.C = C
        Local.TOL = TOL

        Local.tau = np.cos((2*K - 1 - 2*np.arange(K)) * np.pi / (2*K))

        ## Prepare left and right spectral integration matrix, in the position basis
        # TODO faster? when K is small it doesn't matter
        Local.M_toCheb = np.array(
            [[(2 if i > 0 else 1)/K * np.cos(i * (2*K-2*j-1) * np.pi / (2*K))
            for j in range(K)] for i in range(K)]
        )

        M_fromCheb = np.array(
            [[np.cos(j * (2*K-2*i-1) * np.pi / (2*K))
            for j in range(K)] for i in range(K)]
        )

        # Integration matrices in the Chebyshev basis
        M_Ileft = np.zeros((K,K))
        M_Iright = np.zeros((K,K))
        for k in range(1,K-1):
            M_Ileft[k, k-1] = 1/(2*k)
            M_Ileft[k, k+1] = -1/(2*k)
            M_Iright[k, k-1] = -1/(2*k)
            M_Iright[k, k+1] = 1/(2*k)
        M_Ileft[1,0] = 1
        M_Ileft[K-1,K-2] = 1/(2*(K-1))
        M_Iright[1,0] = -1
        M_Iright[K-1,K-2] = -1/(2*(K-1))
        M_Ileft[0,:] = np.sum(
            (-1)**np.arange(K-1).reshape((K-1,1)) * M_Ileft[1:,:], axis=0)
        M_Iright[0,:] = -np.sum(M_Iright[1:,:], axis=0)

        # Integration matrices
        Local.IL = M_fromCheb @ M_Ileft @ Local.M_toCheb
        Local.IR = M_fromCheb @ M_Iright @ Local.M_toCheb

        # Spectral integration
        Local.IC = np.array([2/(1-k*k) if k%2==0 else 0 for k in range(K)]) @ Local.M_toCheb

class IntervalTree:
    """
    Internal representation of the interval tree.
    Note that there is global state about what equation
    it is currently solving, so one cannot solve two
    equations simultaneously.
    """

    f, psi_l, psi_r, g_l, g_r = None, None, None, None, None
    def __init__(self, l:float, r:float, parent : tuple["Branch", bool] = None):
        self.l = l
        self.r = r
        self.alpha_l = self.beta_l = self.delta_l = None
        self.alpha_r = self.beta_r = self.delta_r = None
        # self.lmbda = None
        self.lmbda_l = self.lmbda_r = None
        self.parent = parent

    def upwards(self) -> bool:
        """
        Recursively updates alpha, beta and delta.
        Returns whether it has changed.
        """
        raise NotImplementedError  # Implemented by the two concrete subclasses

    def downwards_root(self):
        """
        Downwards sweep, assuming this is root.
        """
        # self.lmbda = 1
        self.lmbda_l = self.lmbda_r = 0
        self.downwards()

    def downwards(self):
        raise NotImplementedError

    def sweep_right(self) -> list["Leaf"]:
        """Iterates through the leaves."""
        raise NotImplementedError

    def sweep_left(self) -> list["Leaf"]:
        """Iterates through the leaves."""
        raise NotImplementedError

    def refine(self, S):
        """Refine or merge the interval mesh."""
        raise NotImplementedError

    def double(self):
        """Do a final doubling of every interval."""
        raise NotImplementedError

    def fetch_sigma(self):
        """
        Debugging function: Fetches the solution of sigma stored on leaf nodes.
        """
        raise NotImplementedError

    def solve_sigma(self):
        """
        Directly generate a dense matrix representing the discretization of (24)
        in the paper, and solve it to obtain sigma. This is used to verify the
        correctness of the main algorithm, which solves this equation by divide
        and conquer.
        """
        # integration matrix
        M_LI = np.zeros((0,0), dtype=float)
        V_LI = np.zeros(0, dtype=float)
        M_RI = np.zeros((0,0), dtype=float)
        tau_concat = np.zeros(0, dtype=float)
        for leaf in self.sweep_left():
            M_LI = np.block([
                [M_LI,                       np.zeros((M_LI.shape[0], Local.K))],
                [np.tile(V_LI, (Local.K,1)), Local.IL * (leaf.r-leaf.l)/2]])
            V_LI = np.concatenate([V_LI, Local.IC * (leaf.r-leaf.l)/2])
            tau_concat = np.concatenate([tau_concat, leaf.tau])

        for leaf in self.sweep_right():
            M_RI = np.block([
                [M_RI,                               np.tile(Local.IC * (leaf.r-leaf.l)/2, (M_RI.shape[0], 1))],
                [np.zeros((Local.K, M_RI.shape[1])), Local.IR * (leaf.r-leaf.l)/2]])

        P = np.eye(np.shape(M_LI)[0]) \
            + np.diag(IntervalTree.psi_l(tau_concat)) @ M_LI @ np.diag(IntervalTree.g_l(tau_concat)) \
            + np.diag(IntervalTree.psi_r(tau_concat)) @ M_RI @ np.diag(IntervalTree.g_r(tau_concat))
        return np.linalg.solve(P, IntervalTree.f(tau_concat))


class Leaf (IntervalTree):
    def __init__(self, l, r, parent=None):
        if l >= r:
            raise ValueError("Interval invalid:", l, r)
        super().__init__(l, r, parent)
        self.tau = Local.tau * (r - l) / 2 + (r + l) / 2
        self.Ppsi_l = self.Ppsi_r = self.Pf = self.sigma = None
        self.Jr = self.Jl = None
        self.S = None
        self.dirty = True  # An alternative to the update list

    def local_solve(self):
        """
        Solve the integration problem on the subinterval.
        """
        d = (self.r-self.l)/2
        P = np.eye(Local.K) \
            + np.diag(self.psi_l_eval) @ \
                (Local.IL*d) @ np.diag(self.g_l_eval) \
            + np.diag(self.psi_r_eval) @ \
                (Local.IR*d) @ np.diag(self.g_r_eval)
        arr = np.column_stack((self.psi_l_eval, self.psi_r_eval, self.f_eval))
        solution = np.linalg.solve(P, arr)  # Uses gaussian elim with partial pivot
        # evaluate the inner products
        # TODO more vectorized?
        self.Ppsi_l, self.Ppsi_r, self.Pf = solution.T
        self.alpha_l = Local.IC @ (self.g_l_eval * self.Ppsi_l) * d
        self.alpha_r = Local.IC @ (self.g_r_eval * self.Ppsi_l) * d
        self.beta_l = Local.IC @ (self.g_l_eval * self.Ppsi_r) * d
        self.beta_r = Local.IC @ (self.g_r_eval * self.Ppsi_r) * d
        self.delta_l = Local.IC @ (self.g_l_eval * self.Pf) * d
        self.delta_r = Local.IC @ (self.g_r_eval * self.Pf) * d

    def upwards(self):
        if not self.dirty:
            return False
        # cache some results
        self.f_eval, self.psi_l_eval, self.psi_r_eval, self.g_l_eval, self.g_r_eval = IntervalTree.f(self.tau), IntervalTree.psi_l(self.tau), IntervalTree.psi_r(self.tau), IntervalTree.g_l(self.tau), IntervalTree.g_r(self.tau)
        self.local_solve()
        self.dirty = False
        return True
    
    def downwards(self):
        # This is done by my parent
        self.sigma = self.Pf + self.lmbda_l * self.Ppsi_l + self.lmbda_r * self.Ppsi_r

    def sweep_left(self): yield self
    def sweep_right(self): yield self

    def monitor(self):
        """Monitor function, (58) in paper."""
        s_3, s_2, s_1 = Local.M_toCheb[-3:,:] @ self.sigma
        self.S = abs(s_2) + abs(s_3 - s_1)
        return self.S

    def refine(self, S):
        if self.S >= S:
            return self.double()

    def double(self):
        mid = (self.l + self.r) / 2
        new_l = Leaf(self.l, mid)
        new_r = Leaf(mid, self.r)
        new = Branch(new_l, new_r, self.parent)
        if self.parent is None:
            return new
        if self.parent[1]:
            self.parent[0].L = new
        else:
            self.parent[0].R = new

    def fetch_sigma(self):
        return (self.tau, self.sigma)

class Branch (IntervalTree):
    def __init__(self, L:IntervalTree, R:IntervalTree, parent=None):
        if L.l >= R.r : raise ValueError("Interval invalid:", L.l, R.r)
        if L.r != R.l : raise ValueError("Cannot piece interval:", L.r, R.l)
        super().__init__(L.l, R.r, parent)
        self.L = L
        self.L.parent = (self, True)
        self.R = R
        self.R.parent = (self, False)

    def upwards(self):
        # This is head recursive, but if it does get really deep it's Y.P.
        ld = self.L.upwards()
        rd = self.R.upwards()
        if not (ld or rd):
            return False
        # Implements (40)-(45) in paper
        Delta = 1 - self.R.alpha_r * self.L.beta_l
        self.alpha_l = (1 - self.R.alpha_l) * (self.L.alpha_l - self.L.beta_l * self.R.alpha_r) / Delta + self.R.alpha_l
        self.alpha_r = self.R.alpha_r * (1 - self.L.beta_r) * (1 - self.L.alpha_l) / Delta + self.L.alpha_r
        self.beta_l = self.L.beta_l * (1 - self.R.beta_r) * (1 - self.R.alpha_l) / Delta + self.R.beta_l
        self.beta_r = (1 - self.L.beta_r) * (self.R.beta_r - self.L.beta_l * self.R.alpha_r) / Delta + self.L.beta_r
        self.delta_l = (1 - self.R.alpha_l) / Delta * self.L.delta_l + self.R.delta_l + (self.R.alpha_l - 1) * self.L.beta_l / Delta * self.R.delta_r
        self.delta_r = (1 - self.L.beta_r) / Delta * self.R.delta_r + self.L.delta_r + (self.L.beta_r - 1) * self.R.alpha_r / Delta * self.L.delta_l
        return True

    def downwards(self):
        self.L.lmbda_l = self.lmbda_l
        self.R.lmbda_r = self.lmbda_r
        u = self.lmbda_r * (1 - self.R.beta_r) - self.R.delta_r # * self.lmbda
        v = self.lmbda_l * (1 - self.L.alpha_l) - self.L.delta_l # * self.lmbda
        Det = 1 - self.R.alpha_r * self.L.beta_l
        self.L.lmbda_r = (u - self.R.alpha_r*v) / Det
        self.R.lmbda_l = (v - self.L.beta_l*u) / Det
        self.L.downwards()
        self.R.downwards()

    def sweep_left(self):
        yield from self.L.sweep_left()
        yield from self.R.sweep_left()

    def sweep_right(self):
        yield from self.R.sweep_right()
        yield from self.L.sweep_right()

    def refine(self, S):
        if isinstance(self.L, Leaf) and isinstance(self.R, Leaf) and self.L.S + self.R.S < S / 2**Local.K:
            # Merge siblings
            # Note that this only merges siblings, not adjacent cousins.
            new = Leaf(self.l, self.r, self.parent)
            if self.parent is None:
                return new
            if self.parent[1]:
                self.parent[0].L = new
            else:
                self.parent[0].R = new
        else:
            self.L.refine(S)
            self.R.refine(S)

    def double(self):
        self.L.double()
        self.R.double()

    def fetch_sigma(self):
        t1, s1 = self.L.fetch_sigma()
        t2, s2 = self.R.fetch_sigma()
        return (np.concatenate((t1, t2)), np.concatenate((s1, s2)))
