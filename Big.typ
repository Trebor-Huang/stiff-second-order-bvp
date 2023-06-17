#set raw(lang: "python")

#align(center, upper({
    text(size: 15pt, weight: 700, [Course Project Technical Report])
}))

#outline(indent:1em)

= Implementation

We implement the library in Python, with three main components:
- A data structure for managing the interval tree,
- A solver that manipulates the interval tree and monitors for mesh refinement and convergence.
- A non-linear solver on top of the linear one using Newton iteration.
Usage and source code can be found in #link("https://github.com/Trebor-Huang/stiff-second-order-bvp", underline[the GitHub repository]). Some notable implementation details are discussed below.

== Interval Tree

The interval tree is represented as a binary tree `class IntervalTree`. Standard recursion and traversal functionalities are implemented. Each node records the relavant coefficients $alpha_[l,r], beta_[l,r], delta_[l,r]$ and $lambda_[l,r]$. (Note that $lambda$ in @lee is always equal to $1$, and is introduced for convenience of proof only.) The leaf nodes in addition stores arrays representing the discretization of $P^(-1)f, P^(-1) psi_[l,r]$ and $sigma$, which is vital for avoiding duplication of work.

The local solver is implemented by the `class Leaf`. We do not make use of fast Fourier transform algorithms, but instead construct the matrices manually in the initialization phase. This is based on the consideration that the order of the local solver (denoted $K$ in @lee) is usually low and fixed, for instance $K = 16$ in most of the demonstrations in @lee. Using fast Fourier transform may instead perform worse at this scale.

Instead of an `update` list as written in @lee, we mark each leaf node whether it is "dirty", i.e. should be updated when sweeping upwards. This allows for a more structural approach to organizing the program. Other parts of the algorithm are similarly streamlined. For example, the local solution step and the upwards sweep step are integrated in a single recursion.

The description in @lee is ambiguous on the matter of interval merging. We only implemented the merging of sibling intervals, i.e. two leaf nodes that share their parent, instead of more general cousin intervals. Merging adjacent cousin intervals may result in better adaptivity, but seems to induce significant amounts of extra computation, since merging far removed cousins invalidates all the previously computed coefficients. The diagrams in @lee apparently only involves sibling merging, in contrast to the textual specification.

== Linear Solver

The linear solver first performs the appropriate reduction from the equation $ u'' + p(x) u' + q(x) = f(x), x in [l,r] $ $ zeta_(l,0) u(l) + zeta(l,1) u'(l) &= Gamma_l \ zeta_(r,0) u(r) + zeta(r,1) u'(r) &= Gamma_r $ to an integral equation. We first postulate a quadratic function $u_i(x) = a x^2 + b x + c$ satisfying the boundary conditions. Note that the quadratic term is required only when $zeta_([l,r],0) = 0$, at which point $c$ is irrelevant. However for a more unified appoach and for better numeric behavior, we introduce all three parameters in all cases. This is underdetermined, and we pick the solution with the minimal norm for optimal numeric accuracy. The functions $g_l, g_r$ and $psi_l, psi_r$ is then readily constructed as described in the paper.

Given an interval tree, we implemented a direct algorithm that generates (in an unoptimized way) the discretized matrix $overline(P)$ for the tree, and computes $overline(sigma) = overline(P)^(-1) tilde(f)$ by `numpy.linalg.solve`, which uses Gaussian elimination with partial pivoting. This is used to compare against the main algorithm for accuracy and speed.

For the divide-and-conquer algorithm, we perform recursion on the interval tree. This produces $overline(sigma)$ on each leaf interval. The solver then reconstructs $u$ and $u'$ at the discretization points.

@lee suggests a stopping criterion $ ||u_1 - u_0|| / ||u_1 + u_0|| < "Tolerance", $
where $u_1, u_0$ are the last two approximations of the solution $u$. However, since the mesh is different for these two functions, the evaluation is not straightforward. We choose to use a simple linear interpolation scheme, and compute the $L^oo$ norm. After the criterion is satisfied, we finally perform a split of all the leaf intervals, and directly return the result computed with this mesh. Note that the pseudocode description in @lee then continues the original refinement process, so that it goes through three stages of refinement -- doubling -- refinement, we do not implement this logic.

== Newton Interation

Consider the non-linear operator $P u(x) = u''(x) - f(x, u(x), u'(x))$. We need to solve the equation $P u = 0$, with boundary conditions. This operator is Fréchet differentiable, with
$ P(u + delta u) - P(u) = delta u'' - f_2(x, u, u') delta u - f_3(x, u, u') delta u' + o(||delta u||) $
where $f_2, f_3$ are the partial derivatives of $f$. This enables us to use Newton's iteration method. At each stage, we solve the linear differential equation
$ v'' - f_2(x,u,u') v - f_3(x,u,u') v' = P u, $
and we set the next approximation $u$ to be $u - v$. We start the iteration by the quadratic function obtained from the boundary conditions, similar to the linear case.

However, the mesh of each summand is usually different due to the adaptive nature of the algorithm. We choose to keep a list of summands $v_k$ and the corresponding mesh ${x_i}_k$. Each time we use linear interpolation (which is fast and simple) to evaluate the sum.

In addition, we need a way to evaluate the derivatives $u'' = sum_k v_k''$. Fortunately, our algorithm already produces $v_k$ and $v_k'$, and since $v_k$ satisfies the differential equation, we can easily obtain $v_k''$ by subtraction. This avoids evaluating the derivative by finite difference, improving the accuracy.

= Evaluation

We reproduce several test cases from @lee and @starr.

== Linear Solver

We first demonstrate the refinement process using the viscous shock equation $ epsilon u''(x) + 2 x u'(x) = 0. $
With the boundary conditions $u(-1) = -1, u(1) = 1$, the solution rapidly jumps from $-1$ to $1$ near the origin. The following figure shows the convergence process. We take $epsilon = 10^(-5)$.
#align(center, image("Viscous-Shock-Steps.png") + image("Viscous-Shock-Error.png"))
Here the vertical lines mark the left boundaries of the subintervals. This exhibits a similar behavior as in @lee. Note that $"erf"^(-1)$ in the paper refers to the reciprocal, not the functional inverse. We also plot the errors against the refinement stages, where the last stage is when we double all the intervals.

We compare the time used to compute $sigma$ below, using the direct method (i.e. generate the dense matrix and do Gaussian elimination) and the fast method (divide and conquer as described in the paper). We repeat the computation three times and take average.
#align(center, image("Viscous-Shock-Time.png"))
The vertical axis is time measured in seconds, and the horizontal axis is the refinement stages. One can see that there is some initial overhead for the fast method, but it is several orders of magitude faster in later stages. Note that the vertical axis is on a logarithmic scale.

Since the computation is very fast on a modern computer, to accurately measure the time breakdown as in @lee Figure 8, we have to switch to a more computation-heavy task. We turn to the Bessel equation:
$ u'' + (u')/x^2 + (1 - nu^2/x^2)u = 0. $
Subject to the condition $nu = 100, x in [0, 600], u(0) = 0, u(600) = 1$. This transitions from exponentially smooth in $[0,100]$ to oscillating quickly in $[100, 600]$. The algorithm handles it without breaking a sweat.
#align(center, image("Bessel-Solution.png"))
We make a crude time measurement using the module `cProfile`. The total time used was 8.034 seconds. Note that the initialization and termination check phases are too small to draw on the chart. The up/down sweep takes up much more time than in @lee. After investigating with more detailed profiling, it is found that because our restructure of program took a large part of miscellaneous overhead into the recursive sweeping, under the accounting of `cProfile`.
#align(center, image("Bessel-Breakdown.png", width: 50%))
The raw profiling output is included in the repository.

== Non-linear Solver

Since we only implemented the one variable case, we choose the Jacobi elliptic sn function for testing the non-linear solver. Take $m = 0.5$ in $ u'' = 2m u^3 - (1+m) u, $ with boundary condition $u(0) = 0, u(K(m)) = 1$, where $K$ is the elliptic $K$ function. We compare our numerical solution against the analytic solution $"sn"(x, m)$.
#align(center, image("Nonlinear-Solution.png"))
The left is the solution obtained, and the right is the error. There are slightly larger errors at the two boundary nodes, but they do not accumulate and spill as in the finite difference method. Also, note that the resulting mesh is extremely dense, since we merge all the intermediate mesh when outputting. This particular computation took 5 iterations:
#align(center, table(columns: (auto, auto),
    [$||v_k||$], [Time (seconds)],
    [$0.604$], [0.4572],
    [$0.133$], [0.1414],
    [$0.012$], [0.0757],
    [$8.90 times 10^(-5)$], [0.0059],
    [$3.75 times 10^(-8)$], [0.0037],
))

Here we have to turn the tolerance down to $3 times 10^(-5)$, otherwise the mesh refinement seems to take forever. This is because the later iterates have extremely small $||v_k||_oo$, and the stopping criterion uses the relative error, not the absolute one, and therefore likely never reaches termination due to floating point errors. It is possible to alter the termination criterion so that it uses the norm of the current approximation of the _non-linear_ equation in the denominator instead, but this would be breaking the modularity.

= Theoretical Considerations

== Rank Structure of $overline(P)$

There are occasions where the discretized matrix $overline(P)$ is singular. Consider the following equation: $ u''(x) + u(x) = 0, x in [0, pi]. $ The only solutions of this equation are $u = A cos(x + phi)$, thus fixing $u(0) = u(pi)$. If we set $u(0) = 0, u(pi) = 1$, the program does not yield a solution. The discretization quickly converged to a singular matrix. In the divide-and-conquer method, this manifested as division by zero when sweeping upwards. Interestingly, the algorithm coped fine when the condition $u(0) = u(pi)$ is met.

Continuing this line, it is trivial to construct differential equations attacking our specific discretizing scheme, so that $overline(P)$ is exactly singular. Thus not much can be guaranteed about the rank structure of $overline(P)$. However, if the boundary conditions permit a solution, and the Gaussian elimination subroutine can deal with rank-deficient equations, the divide-and-conquer method does yield (one of) the correct results.

== Initial Guess of the Newton Iteration

For the initial guess, we have a quadratic function $ u(x) = a x^2 + b x + c $ that satisfies the boundary conditions. This leaves one more degree of freedom. In the linear case, we choose the solution that minimizes $sqrt(a^2+b^2+c^2)$ so that later computation produces less round-off error. In the non-linear case, we can alternatively try to minimize $||u'' - P u||_oo$ on a simple uniform mesh. This is a non-linear optimization problem over (effectively) one variable. However, the benefit it provides does not compensate the cost.

== Degenerate Boundary Conditions

For degenerate boundary conditions considered in @starr, we are essentially required to solve two problems:
- Given square matrices $A,B,C,D$, find non-singular matrices $X, Y$ such that $A X B + C Y D$ and $A X + C Y$ are both non-singular.
- Find a smooth matrix-valued function that interpolates between $X, Y$ and stays non-singular.

We handle the simpler case where $B = D$ are the identity matrix. This can be dealt with by computing an orthonormal basis for the left null space of $A$ and $C$, then we may either give up due to insufficient rank, or find a rotation such that the two sets of orthonormal vectors (after removing excess ones) combine to form an orthonormal basis of the whole space. These can be taken care of by standard subroutines.

To interpolate between non-singular matrices, we may form the LU-decomposition. Since we are free to invert the sign of columns, we can arrange so that the diagonal entries all have the same sign. Now a simple linear interpolation suffices. This is relatively efficient, and is guaranteed to not run into singularities.

= Conclusion

We implemented a numerical library capable of solving stiff second order boundary value problems over one variable. We tested the library on several examples for speed, accuracy, and investigated its adaptive behavior. The results are compared against the literature. Finally, we considered some related theoretical questions.

The library however is far from perfect. The library only works with equations of one variable, and we would need better error reporting capabilities for users to treat the library as a black box. We used Python for its renowned advantage in fast prototyping, but for practical application, reimplementing the algorithm in languages such as C++ can reduce language-based overhead. In addition, we have not optimized or benchmarked the memory usage in our data structures.

#bibliography("Big.bib")
