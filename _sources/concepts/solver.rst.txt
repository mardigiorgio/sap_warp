Solver
======

:class:`~sim.solver_sap.SolverSAP` is the SAP-native timestepper. It requires a
:class:`~sim.sap_runtime.SapModel` with at least one articulated joint DOF, a
populated :class:`~sim.sap_runtime.SapContacts` buffer, a
:class:`~sim.sap_runtime.SapState` input, a
:class:`~sim.sap_runtime.SapState` output, a
:class:`~sim.sap_runtime.SapControl` object, and a positive timestep.

Timestep Pipeline
-----------------

SAP Warp follows the formulation from `An Unconstrained Convex Formulation of
Compliant Contact`_ and uses `Drake`_'s implementation as a reference. At a
high level, each timestep computes free motion, assembles contact, actuation,
drive, and limit terms, solves the velocity-level SAP problem, and integrates
the resulting state.

Presets
-------

``contact_preset_variant`` should be read as a discretization choice, not as a
different solver. The same timestep is always organized around a free-motion
velocity, a dynamics metric, contact Jacobians, smooth contact potentials,
actuation terms, drive terms, and limit terms. A preset chooses how those
objects are represented.

For preset :math:`P`, the velocity solve can be written as

.. math::

   v_{n+1}^{(P)}
   =
   \arg\min_v
   \ell_P(v),
   \qquad
   \ell_P(v)
   =
   \frac{1}{2}(v-v_P^*)^T A_P(v-v_P^*)
   + \sum_i \ell_i(J_{i,P}v;\,w_{i,P},\mu_i,k_i,\tau_i)
   + \ell_{\mathrm{act},P}
   + \ell_{\mathrm{drive},P}
   + \ell_{\mathrm{limit},P}.

The next configuration is then

.. math::

   q_{n+1}^{(P)}
   =
   \Phi_P(q_n,v_n,v_{n+1}^{(P)},h).

The three built-in presets are coherent choices for
:math:`(v_P^*,A_P,J_{i,P},w_{i,P},\Phi_P)`:

.. list-table::
   :header-rows: 1

   * - Preset
     - Use
     - Main defaults
   * - ``approx32``
     - Fast mixed-precision default for large replicated worlds.
     - fp32 free motion, fp64 contact solve, fp32 linear solve, fp32
       body-inertia contact weights, witness points, midpoint integration.
   * - ``approx64``
     - High-precision ablation of the approximate path.
     - fp64 boundary pose, fp64 free motion, fp64 contact solve, fp64 linear
       solve, fp64 body-inertia contact weights, witness points, midpoint
       integration.
   * - ``drake``
     - Drake-style rigid-contact reference mapping.
     - fp64 boundary pose, fp64 free motion, fp64 contact solve, fp64 linear
       solve, diagonal Delassus contact weights, contact midpoints,
       ``sap_euler`` integration.

The approximate family is

.. math::

   P_{\mathrm{approx}}
   =
   (w_i^{\mathrm{body}},\,J_i^{\mathrm{wit}},\,\Phi_{\mathrm{mid}}),

with ``approx32`` and ``approx64`` differing primarily in floating-point
realization. The Drake-style family is

.. math::

   P_{\mathrm{drake}}
   =
   (w_i^{\mathrm{delassus}},\,J_i^{\mathrm{mid}},\,\Phi_{\mathrm{sap}}).

This distinction is useful in experiments. Compare ``approx32`` to
``approx64`` to measure precision sensitivity,

.. math::

   \Delta_{\mathrm{precision}}
   =
   \|v_{n+1}^{(\mathrm{approx64})}
     - v_{n+1}^{(\mathrm{approx32})}\|,

and compare ``approx64`` to ``drake`` to measure the effect of the contact
metric, contact point, and integration rule,

.. math::

   \Delta_{\mathrm{model}}
   =
   \|v_{n+1}^{(\mathrm{drake})}
     - v_{n+1}^{(\mathrm{approx64})}\|.

Explicit constructor keyword arguments override preset values after the preset
is expanded, which makes one-factor experiments possible. For example,
``contact_preset_variant="approx64"`` with
``contact_weight_mode="diag_delassus"`` changes only the contact-weight model
while leaving the approximate-family contact point and integration map intact.

Preset Modes
------------

Each preset is a named bundle of mode variants. The modes can be read directly
against the objective above.

``contact_weight_mode``
   The effective contact weight controls the metric scale used by the smooth
   contact potential. Through the chain rule, a single contact contributes

   .. math::

      \nabla\ell_i(v)=J_i^T g_i(J_i v;w_i),
      \qquad
      \nabla^2\ell_i(v)=J_i^T G_i(J_i v;w_i)J_i.

   ``body_inertia`` estimates :math:`w_i` from the two contacted bodies. For
   contact-frame directions :math:`C_i=\{c_{t1},c_{t2},c_n\}`, witness
   :math:`x_b`, center of mass :math:`o_b`, body rotation :math:`R_b`, mass
   :math:`m_b`, and diagonal body inertia :math:`I_b`,

   .. math::

      \omega_b(c)
      =
      \frac{1}{m_b}
      +
      \left(R_b^T((x_b-o_b)\times c)\right)^T
      I_b^{-1}
      \left(R_b^T((x_b-o_b)\times c)\right),
      \qquad
      w_i^{\mathrm{body}}
      =
      \max\left(
         \frac{1}{3}
         \sum_{c\in C_i}
         \sum_b \omega_b(c),
         10^{-12}
      \right).

   ``diag_delassus`` estimates a contact-space mobility from the assembled
   dynamics matrix and contact Jacobian. The current implementation uses the
   diagonal dynamics inverse:

   .. math::

      \widehat W_i
      =
      J_i\,\mathrm{diag}(A)^{-1}J_i^T,
      \qquad
      w_i^{\mathrm{delassus}}
      =
      \max\left(\frac{\|\widehat W_i\|_F}{3},10^{-12}\right).

``contact_point_mode``
   The collision pipeline supplies two witnesses :math:`x_0,x_1`, a normal,
   margins, and material data. ``witness_point`` uses those witnesses directly:

   .. math::

      J_i^{\mathrm{wit}}v
      =
      R_{WC}^T\left[V_1(x_1;v)-V_0(x_0;v)\right].

   ``contact_midpoint`` first computes a stiffness-weighted point

   .. math::

      p_C=\alpha_0x_0+\alpha_1x_1,
      \qquad
      \alpha_0=\frac{k_0}{k_0+k_1},
      \qquad
      \alpha_1=\frac{k_1}{k_0+k_1},

   and evaluates both body velocities at :math:`p_C`:

   .. math::

      J_i^{\mathrm{mid}}v
      =
      R_{WC}^T\left[V_1(p_C;v)-V_0(p_C;v)\right].

``position_integration``
   The position map runs after the velocity minimization. ``midpoint`` uses the
   incoming and solved SAP velocities,

   .. math::

      \bar v = \frac{v_n+v_{n+1}}{2},
      \qquad
      q_{n+1}=\Phi_{\mathrm{mid}}(q_n,\bar v,h),

   while ``sap_euler`` uses the solved velocity:

   .. math::

      q_{n+1}=\Phi_{\mathrm{sap}}(q_n,v_{n+1},h).

Precision Modes
   ``free_motion_solve_precision``, ``contact_solve_precision``,
   ``contact_linear_solve_precision``, ``sap_contact_weight_precision``, and
   ``use_f64_boundary_pose`` control where fp32 or fp64 arrays are used. A
   value computed through fp32 and then stored in fp64 should be understood as

   .. math::

      x^{(32\rightarrow64)}
      =
      \mathrm{cast}_{64}(\mathrm{fl}_{32}(x)).

   The following fp64 contact solve uses this projected value; it does not
   reconstruct the precision that was discarded earlier. This is why
   ``approx64`` is the right comparison point when isolating the effect of
   ``approx32``'s mixed-precision path.

Line Search
-----------

``line_search_variant`` is separate from ``contact_preset_variant``. It controls
how the contact solve accepts a Newton step after the preset has chosen the
contact weights, contact point mode, precision path, and integration mode.

The common one-dimensional problem is

.. math::

   H_k d_k = -\nabla\ell(v_k),
   \qquad
   \varphi_k(\alpha)=\ell(v_k+\alpha d_k),
   \qquad
   v_{k+1}=v_k+\alpha d_k.

The line search evaluates the SAP objective along :math:`\varphi_k`, including
the nonlinear contact, drive, and limit regularizers at the trial velocity.

``monotone_decay``
   Conservative default. It tests :math:`\alpha=1,\frac{1}{2},\frac{1}{4},\ldots`
   and accepts the first candidate satisfying

   .. math::

      \varphi_k(\alpha)
      \le
      \varphi_k(0)
      + 10^{-14}
      + 10^{-12}|\varphi_k(0)|.

   It is the cheapest and most predictable path because it needs only trial
   costs and a fixed geometric decay.

``armijo_decay``
   Armijo-style backtracking controlled by ``armijo_c``, ``rho``, and
   ``line_search_relative_slop``. It first probes
   :math:`\alpha_{\max}=1/\rho` and can accept that over-relaxed step if the
   line derivative is still negative or only slightly positive within the slop.
   If not, it tests :math:`1,\rho,\rho^2,\ldots`. The sufficient-decrease test is

   .. math::

      \varphi_k(\alpha)
      \le
      \varphi_k(0) + c\,\alpha\,\varphi_k'(0).

``exact_root``
   Root-search variant. It brackets the stationary point on
   :math:`0\le\alpha\le1.5` and solves

   .. math::

      \varphi_k'(\alpha)
      =
      \frac{d}{d\alpha}\ell(v_k+\alpha d_k)
      =
      0.

   It uses Newton updates when they stay inside the bracket and bisection when
   they do not. If the default maximum line-search count is still ``40``, the
   solver raises it to ``100`` for this variant.

Convergence controls include ``max_iterations``, ``optimality_abs_tol``,
``optimality_rel_tol``, ``cost_abs_tol``, ``cost_rel_tol``,
``line_search_max_iterations``, and ``line_search_relative_slop``. The solver
also exposes ``last_line_search_iterations`` for profiling accepted steps.

Objective Summary
-----------------

SAP advances in two stages. First, free motion predicts the unconstrained
velocity ``v_star``. Second, the solver minimizes a convex velocity objective
that keeps the solution close to free motion in the dynamics metric while
adding smooth terms for contact, friction, actuation, joint drives, and limits.

Solver Configuration and Presets
--------------------------------

The solver surface is :class:`~sim.solver_sap.SolverSAP`. It requires a
:class:`~sim.sap_runtime.SapModel` with articulated joint DOFs and does not
support gradient states. ``benchmark.py`` constructs it directly from the
loaded scene:

.. code-block:: python

   solver = SolverSAP(
       loaded.sap_model,
       max_rigid_contact=max_rigid_contact,
       contact_tau_d=scene_default_shape_tau,
       **simulation_solver_kwargs,
   )

Pipeline
~~~~~~~~

:meth:`~sim.solver_sap.SolverSAP.step` performs four stages:

1. Convert public boundary state/control arrays into the solver's SAP ordering
   when needed.
2. Run free motion through :class:`~sim.free_motion.SapFreeMotion`.
3. Build contact Jacobians and per-environment dynamics matrices through
   :class:`~sim.contact_jacobian.SapContactJacobian`.
4. Solve the stage-2 SAP velocity problem through
   :class:`~sim.contact_solve.SapContactSolve` and integrate ``state_out``.

Collision generation is separate. The caller is expected to run
:meth:`~sim.collision.pipeline.SapCollisionPipeline.collide` and pass the
resulting contacts into :meth:`~sim.solver_sap.SolverSAP.step`.

Blocked Cholesky
~~~~~~~~~~~~~~~~

The blocked Cholesky kernels used for the Newton direction are based on
`NVIDIA Warp's tile blocked-Cholesky example
<https://github.com/NVIDIA/warp/blob/main/warp/examples/tile/example_tile_block_cholesky.py>`__
code and adapted for SAP Warp's batched, masked, and multi-right-hand-side
contact solve paths. The surrounding solver assembly, precision switches, and
contact objective are SAP Warp code.

Contact Presets
~~~~~~~~~~~~~~~

``contact_preset_variant`` chooses a coherent numerical discretization of the
same SAP timestep. It is best read as a small experimental design: each preset
selects the precision used by the free-motion and contact stages, the formula
used for the effective contact weight, the point at which the contact Jacobian
is evaluated, and the map that integrates the solved velocity back to
generalized position.

All presets still assemble a stage-2 velocity objective of the form

.. math::

   \ell_P(v)
   =
   \frac{1}{2}(v - v^*_P)^T A_P (v - v^*_P)
   + \sum_{i=1}^{n_c} \ell_i(J_{i,P} v;\, w_{i,P}, \mu_i, k_i, \tau_i)
   + \ell_{\mathrm{act},P}(v)
   + \ell_{\mathrm{drive},P}(v)
   + \ell_{\mathrm{limit},P}(v),

where :math:`P` denotes the selected preset. The minimizer is then integrated by
the preset's position map:

.. math::

   v_{n+1}^{(P)} = \arg\min_v \ell_P(v),
   \qquad
   q_{n+1}^{(P)} = \Phi_P(q_n, v_n, v_{n+1}^{(P)}, h).

Thus ``approx32``, ``approx64``, and ``drake`` are not different contact laws.
They are different approximations to the same discrete variational problem. The
important question is which quantities in :math:`\ell_P` are evaluated cheaply,
which are evaluated in reference precision, and which geometric point defines
the relative contact velocity.

.. list-table::
   :header-rows: 1

   * - Preset
     - Intent
     - Precision path
     - Contact modes
     - Integration
   * - ``approx32``
     - Fast mixed-precision default for large batches.
     - f32 boundary pose, fp32 free motion, fp64 contact solve, fp32 linear
       solve, fp32 contact weights.
     - ``body_inertia`` contact weights with ``witness_point`` contact points.
     - ``midpoint``.
   * - ``approx64``
     - Precision ablation of the approximate path.
     - f64 boundary pose, fp64 free motion, fp64 contact solve, fp64 linear
       solve, fp64 contact weights.
     - ``body_inertia`` contact weights with ``witness_point`` contact points.
     - ``midpoint``.
   * - ``drake``
     - Drake-style reference mapping for the rigid-contact path.
     - f64 boundary pose, fp64 free motion, fp64 contact solve, fp64 linear
       solve, fp64 contact weights.
     - ``diag_delassus`` contact weights with ``contact_midpoint`` contact
       points.
     - ``sap_euler``.

If omitted, :class:`~sim.solver_sap.SolverSAP` uses ``approx32``.
``approx_32``/``approx-32`` and ``approx_64``/``approx-64`` are normalized to
``approx32`` and ``approx64``.

Preset Matrix
^^^^^^^^^^^^^

The table above is compact. The following matrix is the more useful way to
read the implementation. Each row names a mathematical object in the objective
and shows how the presets choose it.

.. list-table::
   :header-rows: 1

   * - Object
     - ``approx32``
     - ``approx64``
     - ``drake``
   * - Boundary pose used during contact preparation
     - Public pose converted through the f32 path.
     - Boundary pose promoted to f64.
     - Boundary pose promoted to f64.
   * - Free-motion solve :math:`v^*_P`
     - fp32 linear algebra.
     - fp64 linear algebra.
     - fp64 linear algebra.
   * - Contact objective buffers
     - fp64 scalar objective and derivative buffers.
     - fp64 scalar objective and derivative buffers.
     - fp64 scalar objective and derivative buffers.
   * - Newton linear solve :math:`H_k d_k=-\nabla\ell(v_k)`
     - fp32 blocked Cholesky path.
     - fp64 blocked Cholesky path.
     - fp64 blocked Cholesky path.
   * - Effective contact weight :math:`w_{i,P}`
     - fp32 body-inertia estimate.
     - fp64 body-inertia estimate.
     - fp64 diagonal-Delassus estimate.
   * - Contact Jacobian :math:`J_{i,P}`
     - Relative velocity at the two collision witnesses.
     - Relative velocity at the two collision witnesses.
     - Relative velocity at a stiffness-weighted contact midpoint.
   * - Position map :math:`\Phi_P`
     - Midpoint velocity integration.
     - Midpoint velocity integration.
     - SAP Euler integration.

This layout gives the presets a useful interpretation:

``approx32``
   Optimizes throughput and memory traffic while keeping the contact objective
   itself in fp64. It is the default because large replicated worlds are often
   limited by the free-motion path, contact-weight preparation, and the contact
   Newton linear solve.

``approx64``
   Keeps the approximate geometry and weight model but removes the deliberate
   fp32 choices. In practice it is the preset to use when a scene behaves
   differently under ``approx32`` and you want to know whether the difference is
   primarily numerical precision or the approximate contact model.

``drake``
   Uses the Drake-style rigid-contact discretization: f64 throughout the SAP
   path, a Delassus-based effective weight, midpoint contact geometry, and SAP
   Euler position integration. This is the preset to use when comparing against
   a reference SAP formulation or when stiff contact stacks are more important
   than maximum throughput.

The contact solve accepts explicit keyword overrides after the preset has been
expanded. For example, using ``contact_preset_variant="approx64"`` and then
setting ``contact_weight_mode="diag_delassus"`` creates a single-mode
experiment: precision and integration stay approximate-family, while the weight
model changes.

Approx32
^^^^^^^^

``approx32`` is the fast approximate member of the family. It chooses the
body-inertia contact weight, witness-point contact Jacobian, midpoint position
integration, fp32 free motion, fp32 contact weights, and an fp32 Newton linear
solve. The scalar contact objective still runs in fp64, so the contact
potential, gradient, and Hessian contributions are evaluated in the same scalar
precision as the other presets once the approximate data have been assembled.

For a contact :math:`i`, let
:math:`C_i=\{c_{t1},c_{t2},c_n\}` be the orthonormal contact frame in world
coordinates. For body :math:`b`, contact witness :math:`x_b`, center of mass
:math:`o_b`, world rotation :math:`R_b`, mass :math:`m_b`, and diagonal body
inertia :math:`I_b`, the directional body mobility used by the implementation
is

.. math::

   \omega_b(c)
   =
   \frac{1}{m_b}
   +
   \left(R_b^T ((x_b-o_b)\times c)\right)^T
   I_b^{-1}
   \left(R_b^T ((x_b-o_b)\times c)\right).

The ``body_inertia`` weight averages these directional mobilities over the two
tangent axes and the normal axis:

.. math::

   w_i^{\mathrm{body}}
   =
   \max\left(
      \frac{1}{3}
      \sum_{c\in C_i}
      \sum_{b\in\{0,1\}\cap \mathrm{dynamic}}
      \omega_b(c),
      \epsilon
   \right),
   \qquad
   \epsilon=10^{-12}.

In ``approx32`` this expression is evaluated through the fp32 path before being
stored in the fp64 contact buffers. The approximation is intentionally local:
it uses the inertia of the bodies that own the two shapes and the contact
lever arms, but it does not apply the articulated dynamics matrix
:math:`A_P^{-1}` to the full contact Jacobian. This saves the per-contact
matrix traversal used by the Delassus path.

The contact Jacobian is evaluated at the witness points produced by collision:

.. math::

   J_i^{\mathrm{wit}} v
   =
   R_{WC}^T
   \left[
      V_1(x_1;v) - V_0(x_0;v)
   \right],

where :math:`R_{WC}` maps world velocity into the contact frame and
:math:`V_b(x;v)` is the world velocity of body :math:`b` at point :math:`x`.
When the two witnesses are separated by a small positive gap, this measures
relative velocity at the actual pair of geometric witnesses rather than first
collapsing them to a common point. That is why the preset is called
``approx``: the solve is still a SAP solve, but the contact metric and contact
geometry are less globally coupled than the Drake-style path.

After the velocity solve, position integration uses a midpoint velocity:

.. math::

   \bar v = \frac{v_n + v_{n+1}}{2},
   \qquad
   q_{n+1} = \Phi_{\mathrm{mid}}(q_n,\bar v,h).

For scalar joints this is the familiar
:math:`q_{n+1}=q_n+h\bar v`. For quaternion joints the implementation applies
the corresponding midpoint angular increment and normalizes the result. This
choice is useful for the approximate presets because it damps some visual
discrepancy between the explicit state :math:`q_n` and the solved velocity
:math:`v_{n+1}` when the free-motion path is intentionally lightweight.

Approx64
^^^^^^^^

``approx64`` keeps the same mathematical approximations as ``approx32`` but
changes the numerical precision of the path:

.. math::

   J_i^{(\mathrm{approx64})} = J_i^{\mathrm{wit}},
   \qquad
   w_i^{(\mathrm{approx64})} = w_i^{\mathrm{body}},
   \qquad
   \Phi_{\mathrm{approx64}} = \Phi_{\mathrm{mid}}.

The important difference is that boundary pose preparation, free motion,
contact-weight evaluation, the stage-2 scalar buffers, and the blocked Newton
linear solve all use fp64. This makes ``approx64`` the cleanest diagnostic
preset. If

.. math::

   \Delta v_{\mathrm{round}}
   =
   \|v_{n+1}^{(\mathrm{approx64})}
     - v_{n+1}^{(\mathrm{approx32})}\|

is small for a scene, then the approximate-family choices
``body_inertia``/``witness_point``/``midpoint`` dominate the behavior relative
to ``drake``. If :math:`\Delta v_{\mathrm{round}}` is large, the scene is
sensitive to fp32 free motion, fp32 contact weights, or the fp32 Newton linear
solve. The preset is therefore useful even when it is not the intended runtime
default: it separates floating-point sensitivity from modeling sensitivity.

Because ``approx64`` still uses witness points, its contact Jacobian has the
same geometric support as ``approx32``. Because it still uses body-inertia
weights, its effective contact metric remains local to the contacted bodies.
The preset should therefore not be interpreted as "Drake in fp64"; it is the
high-precision version of the approximate implementation.

Drake
^^^^^

``drake`` selects the reference-style rigid-contact path. It keeps the solver
data in fp64, evaluates the contact Jacobian at a single contact point, computes
the effective weight from a Delassus-style matrix, and integrates position with
the SAP Euler map.

For each contact, the implementation forms a diagonal Delassus estimate from
the assembled per-environment dynamics matrix:

.. math::

   \widehat W_i
   =
   J_i\,\mathrm{diag}(A_P)^{-1}J_i^T,
   \qquad
   w_i^{\mathrm{delassus}}
   =
   \max\left(\frac{\|\widehat W_i\|_F}{3}, \epsilon\right),
   \qquad
   \epsilon=10^{-12}.

The use of :math:`\mathrm{diag}(A_P)^{-1}` is an implementation choice: it
captures the scale of the contact-space mobility while avoiding a dense solve
for every candidate contact. Compared with ``body_inertia``, this weight sees
the contact Jacobian and the assembled generalized dynamics metric, so the
regularization in :math:`\ell_i` is tied more directly to the velocity unknowns
used by the Newton system.

The contact point is a stiffness-weighted midpoint of the two collision
witnesses:

.. math::

   p_C
   =
   \alpha_0 x_0 + \alpha_1 x_1,
   \qquad
   \alpha_0 = \frac{k_0}{k_0+k_1},
   \qquad
   \alpha_1 = \frac{k_1}{k_0+k_1},

with :math:`\alpha_0=\alpha_1=\frac{1}{2}` when the denominator is zero. The
Jacobian then measures both body velocities at :math:`p_C`:

.. math::

   J_i^{\mathrm{mid}} v
   =
   R_{WC}^T
   \left[
      V_1(p_C;v) - V_0(p_C;v)
   \right].

This is closer to the usual point-contact abstraction: the normal gap still
comes from the collision witnesses and margins, but the velocity constraint is
represented at one shared contact point. For stiff contacts this tends to make
the local contact model and the generalized dynamics metric agree more closely.

Finally, SAP Euler integration advances positions using the solved velocity:

.. math::

   q_{n+1}
   =
   \Phi_{\mathrm{sap}}(q_n, v_{n+1}, h),

which reduces to :math:`q_{n+1}=q_n+h\,v_{n+1}` for scalar coordinates and uses
the corresponding SAP quaternion update for rotational coordinates. The result
is a preset whose contact geometry, weight metric, and integration rule match
the intended Drake-style SAP discretization more closely than the approximate
presets.

Mode Variants
^^^^^^^^^^^^^

Each preset is assembled from four independent mode families. Reading them
individually is useful when you override one constructor keyword at a time.

Contact Weight Mode
"""""""""""""""""""

The effective contact weight :math:`w_i` scales the smooth contact potential
for contact :math:`i`. In the Newton system it appears through the contact
gradient and Hessian terms

.. math::

   \nabla\ell_i(v) = J_i^T g_i(J_i v;w_i),
   \qquad
   \nabla^2\ell_i(v) = J_i^T G_i(J_i v;w_i)J_i.

``body_inertia`` is local and cheap. It computes :math:`w_i` from body masses,
body-frame diagonal inertias, contact-frame directions, and witness lever arms,
using the average formula shown in the ``approx32`` section. It is selected by
``approx32`` and ``approx64``.

``diag_delassus`` is dynamics-aware. It first assembles the contact Jacobian
and the per-environment dynamics matrix, then estimates a contact-space
Delassus matrix with :math:`\mathrm{diag}(A)^{-1}`:

.. math::

   \widehat W_i
   =
   J_i\,\mathrm{diag}(A)^{-1}J_i^T,
   \qquad
   w_i
   =
   \max\left(\frac{\|\widehat W_i\|_F}{3},10^{-12}\right).

It is selected by ``drake``.

Contact Point Mode
""""""""""""""""""

``contact_point_mode`` selects where the relative velocity in :math:`J_i v` is
measured. The collision record always stores two witness points :math:`x_0` and
:math:`x_1`, a normal, two margins, and material data. The mode only changes
the velocity point used by the Jacobian.

``witness_point`` evaluates the velocity of each body at its own witness:

.. math::

   J_i v = R_{WC}^T\{V_1(x_1;v)-V_0(x_0;v)\}.

``contact_midpoint`` first computes :math:`p_C` and evaluates both velocities
there:

.. math::

   p_C = \alpha_0 x_0+\alpha_1 x_1,
   \qquad
   J_i v = R_{WC}^T\{V_1(p_C;v)-V_0(p_C;v)\}.

The midpoint mode is paired with the Delassus weight in ``drake`` because both
choices make the contact term depend on a single contact-space velocity
measured against the assembled generalized dynamics.

Position Integration Mode
"""""""""""""""""""""""""

``position_integration`` is applied after the velocity solve. It does not
change the minimization problem already solved for :math:`v_{n+1}`, but it
does change the next configuration used by collision and free motion.

``midpoint`` uses the average of the incoming SAP velocity and the solved SAP
velocity:

.. math::

   \bar v = \frac{v_n+v_{n+1}}{2},
   \qquad
   q_{n+1} = \Phi_{\mathrm{mid}}(q_n,\bar v,h).

``sap_euler`` uses the solved velocity directly:

.. math::

   q_{n+1} = \Phi_{\mathrm{sap}}(q_n,v_{n+1},h).

Precision Modes
"""""""""""""""

The precision knobs choose where fp32/fp64 arrays enter the preset. They do not
rename the physical model; they change the floating-point realization of the
same equations.

.. list-table::
   :header-rows: 1

   * - Knob
     - Mathematical object
     - What changes
   * - ``free_motion_solve_precision``
     - :math:`v^*_P` and the dynamics data prepared by free motion.
     - Precision of the linear algebra used before contacts are applied.
   * - ``contact_solve_precision``
     - :math:`v`, :math:`\nabla \ell`, :math:`H`
     - Scalar precision used by the stage-2 Newton solve.
   * - ``contact_linear_solve_precision``
     - :math:`H d = -\nabla \ell`
     - Precision of the blocked Cholesky linear solve for the Newton direction.
   * - ``sap_contact_weight_precision``
     - :math:`w_i`
     - Precision used to compute the effective contact weight.
   * - ``contact_weight_mode``
     - :math:`w_i`
     - Formula used for the scalar effective weight in contact regularization.
   * - ``contact_point_mode``
     - :math:`J_i`
     - Point where the contact Jacobian measures relative velocity.
   * - ``position_integration``
     - :math:`q_{n+1}`
     - Formula used to integrate positions after :math:`v_{n+1}` is solved.

Precision Knobs
^^^^^^^^^^^^^^^

Each preset can be overridden with explicit constructor kwargs. These are the
same knobs described above, exposed directly for controlled comparisons:

``free_motion_solve_precision``
   ``fp32``/``f32`` or ``fp64``/``f64`` for the free-motion solve that prepares
   :math:`v^*_P`.

``contact_solve_precision``
   Scalar dtype used by the stage-2 contact objective buffers.

``contact_linear_solve_precision``
   Linear solve dtype used by the blocked Cholesky path for
   :math:`H_k d_k=-\nabla\ell(v_k)`.

``sap_contact_weight_precision``
   Precision used while computing :math:`w_i`.

``use_f64_boundary_pose``
   Convert boundary poses to f64 for contact preparation.

When a value is computed through fp32 and later stored in an fp64 buffer, it is
useful to think of a projection

.. math::

   x^{(32\rightarrow64)} = \mathrm{cast}_{64}(\mathrm{fl}_{32}(x)).

The subsequent fp64 contact solve cannot recover information lost by the
earlier projection. This is the main distinction between ``approx32`` and
``approx64``.

Contact Capacity
~~~~~~~~~~~~~~~~

``max_rigid_contact`` is the per-world contact capacity used by
:class:`~sim.solver_sap.SolverSAP` and its env-local contact buffers.
``benchmark.py`` reads
``simulation.max_rigid_contact`` as this per-env cap, then sizes the flat
:class:`~sim.collision.pipeline.SapCollisionPipeline` contact buffer as
``simulation.max_rigid_contact * num_worlds``.

If generated contacts exceed capacity, the extra contacts are dropped and
``solver.last_truncated_contact_count`` records the truncated count reported by
:class:`~sim.contact_jacobian.SapContactJacobian`.

Line Search
~~~~~~~~~~~

``line_search_variant`` is independent from ``contact_preset_variant``. A scene
can use ``contact_preset_variant="drake"`` with ``line_search_variant="armijo_decay"``,
or an approximate preset with ``exact_root``. The line search controls how the
contact solve accepts a Newton direction after the stage-2 objective has been
assembled.

At Newton iteration :math:`k`, define the one-dimensional objective

.. math::

   \varphi_k(\alpha)
   =
   \ell(v_k+\alpha d_k).

The Newton direction is computed from

.. math::

   H_k d_k = -\nabla \ell(v_k),

so a valid active Newton environment should have
:math:`\varphi_k'(0)=\nabla\ell(v_k)^Td_k<0`. The line search chooses
:math:`\alpha` and commits

.. math::

   v_{k+1} = v_k + \alpha d_k.

The implementation evaluates the SAP cost along this line, not only a quadratic
model. The dynamics term is updated from precomputed line coefficients,

.. math::

   m_k(\alpha)
   =
   m_k(0)
   + \alpha\,m'_k(0)
   + \frac{1}{2}\alpha^2 m''_k,

while contact, drive, and limit regularizers are recomputed at the trial
velocity. This matters for contact because the SAP regularizers are smooth but
nonlinear in :math:`J_i(v_k+\alpha d_k)`.

``monotone_decay``
   Conservative default. It tries the sequence

   .. math::

      \alpha_j = 2^{-j},
      \qquad j=0,1,\ldots,

   and accepts the first candidate whose trial cost is monotone up to the
   fixed device-side relaxation

   .. math::

      \varphi_k(\alpha_j)
      \le
      \varphi_k(0)
      + 10^{-14}
      + 10^{-12}\,|\varphi_k(0)|.

   If the step falls below :math:`10^{-8}` before satisfying the test, the
   line search reports a failure status for that environment. This variant does
   not use the directional slope in its acceptance rule, which makes it simple
   and cheap. It is the runtime default because it is stable for batched scenes
   and has predictable work: one fused kernel evaluates the candidate cost,
   accepts the step, and writes the committed velocity.

``armijo_decay``
   Armijo-style backtracking with an over-relaxed first probe. ``rho`` must lie
   in ``(0, 1)`` and ``armijo_c`` controls sufficient decrease. The first trial
   is

   .. math::

      \alpha_{\max} = \frac{1}{\rho},

   so with the default ``rho=0.8`` the solver first tests
   :math:`\alpha_{\max}=1.25`. If the derivative at that point is still
   negative, or only positive within the line-search slop, the over-relaxed
   step is accepted. Otherwise the candidate sequence becomes

   .. math::

      \alpha_{\max},\; 1,\; \rho,\; \rho^2,\ldots .

   A candidate is accepted using the Armijo condition

   .. math::

      \varphi_k(\alpha)
      \le
      \varphi_k(0) + c\,\alpha\,\varphi_k'(0),

   together with a finite-difference slope check between consecutive candidate
   costs. If both the current and previous candidates satisfy Armijo near the
   bracketed turn in the line, the previous larger step can be selected. The
   ``line_search_relative_slop`` parameter defaults to
   :math:`1000\,\epsilon_{\mathrm{dtype}}` and is scaled by the local cost
   magnitude before these near-flat tests are applied.

``exact_root``
   Exact-root search path. If ``line_search_max_iterations`` is left at the
   constructor default ``40``, the solver raises it to ``100`` for this variant.
   It searches for the stationary point of :math:`\varphi_k` along the Newton
   direction:

   .. math::

      \varphi_k'(\alpha)
      =
      \frac{d}{d\alpha}\ell(v_k + \alpha d_k)
      =
      0.

   The search interval is

   .. math::

      0 \le \alpha \le \alpha_{\max},
      \qquad
      \alpha_{\max}=1.5.

   If :math:`\varphi_k'(\alpha_{\max})\le 0`, the cost is still decreasing at
   the upper end and the solver accepts :math:`\alpha_{\max}`. Otherwise it
   brackets the root between :math:`0` and :math:`\alpha_{\max}` and starts from
   the Newton estimate

   .. math::

      \alpha_0
      =
      \mathrm{clip}
      \left(
         \frac{-\varphi_k'(0)}{\varphi_k''(\alpha_{\max})},
         0,
         \alpha_{\max}
      \right).

   Each iteration tries a Newton update, but falls back to bisection whenever
   the Newton point leaves the bracket, becomes non-finite, or is judged too
   slow. The normalized derivative tolerance is :math:`10^{-8}`. This variant
   evaluates the trial derivative and second derivative, including contact
   Hessian terms, so it is more expensive than the decay variants. It is useful
   for solver comparisons and stiff scenes where the accepted step length is
   important enough to justify the additional work.

Convergence and diagnostic controls:

``line_search_max_iterations``
   Maximum number of candidate line-search iterations. ``exact_root`` promotes
   the constructor default from ``40`` to ``100``.

``armijo_c`` and ``rho``
   Armijo sufficient-decrease coefficient and geometric decay factor for
   ``armijo_decay``.

``line_search_relative_slop``
   Near-flat tolerance used by the Armijo-style path. If unset, it is
   :math:`1000\,\epsilon_{\mathrm{dtype}}`.

``cost_abs_tol`` and ``cost_rel_tol``
   Outer Newton cost-convergence tolerances. If unset, ``monotone_decay`` uses
   ``0.0`` and ``5.0e-3``; ``armijo_decay`` and ``exact_root`` use
   ``1.0e-30`` and ``1.0e-15``. ``exact_root`` also uses these values to accept
   a unit step when the initial directional decrease is already below the cost
   tolerance.

``last_line_search_iterations``
   The solver records the total accepted line-search iterations from the last
   step for profiling and comparisons.

.. _An Unconstrained Convex Formulation of Compliant Contact: https://arxiv.org/abs/2110.10107
.. _Drake: https://github.com/RobotLocomotion/drake
