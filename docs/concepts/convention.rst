Convention
==========

SAP Warp uses two closely related conventions at the solver boundary. The
``public`` convention is the one scene loaders, benchmark loops, and user code
should normally see. The ``sap`` convention is the angular-first,
body-origin convention consumed by the SAP kernels.

Generalized Position Storage
----------------------------

``joint_q`` is a packed generalized-position array. Joint ``j`` stores its
position coordinates starting at ``joint_q_start[j]``; the next entry in
``joint_q_start`` gives the end of the segment. These position coordinates have
fixed storage and do not have a ``public``/``sap`` order flag.

Quaternions are stored in Warp's ``xyzw`` layout:

.. code-block:: text

   [x, y, z, w]

The per-joint ``joint_q`` segment is:

.. list-table::
   :header-rows: 1

   * - Joint type
     - ``joint_q`` segment
     - Meaning
   * - ``prismatic``
     - ``[q]``
     - Translation along the joint axis.
   * - ``revolute``
     - ``[q]``
     - Rotation angle about the joint axis, in radians.
   * - ``ball``
     - ``[q_x, q_y, q_z, q_w]``
     - Quaternion rotation.
   * - ``fixed``
     - empty
     - No position coordinates.
   * - ``free`` / ``distance``
     - ``[p_x, p_y, p_z, q_x, q_y, q_z, q_w]``
     - Translation followed by quaternion rotation.
   * - ``D6`` / general multi-axis
     - ``[q_l0, ..., q_lN, q_a0, ..., q_aM]``
     - Linear-axis coordinates first, then angular-axis coordinates, in the
       model's stored axis order.

For example, free-joint positions store translation followed by the quaternion:

.. code-block:: text

   [p_x, p_y, p_z, q_x, q_y, q_z, q_w]

Ball-joint positions store only the quaternion:

.. code-block:: text

   [q_x, q_y, q_z, q_w]

The quaternion storage order is independent of velocity ordering. For example,
``sap`` velocity order for a free joint is angular-first, but the free-joint
position still stores translation first and quaternion second. During
integration, the solver maps the solved SAP velocity back through the joint
kinematics before writing ``state_out.joint_q`` and ``state_out.joint_qd``.

Consequently, ``joint_q`` and ``joint_qd`` do not always have the same segment
length. Ball joints store four position scalars but have three velocity DOFs;
free and distance joints store seven position scalars but have six velocity
DOFs.

Generalized Velocity and Force Storage
--------------------------------------

The distinction matters only for joints whose generalized velocity represents a
free rigid-body twist: free joints and distance joints. Scalar joints
(``revolute`` and ``prismatic``), ball joints, fixed joints, and general
multi-axis joints keep their stored DOF order. They may still be copied into
temporary fp64 buffers, but they do not need the reference-point transformation
derived below.

Unchanged DOF Segments
~~~~~~~~~~~~~~~~~~~~~~

For these joints, the public boundary adapter does not reorder the per-DOF
arrays. Entry ``k`` in a joint's ``joint_qd`` segment is copied to entry ``k``
in the SAP buffer, and the matching ``joint_f``, target, limit, and armature
entries keep the same meaning.

Inside the solver, each scalar DOF is then expanded into a spatial-vector
column using the joint axis and current pose. That internal spatial vector uses
SAP's angular-first layout, but this expansion does not change the order of the
packed per-DOF arrays.

.. list-table::
   :header-rows: 1

   * - Joint type
     - Stored DOF segment
     - Boundary treatment
     - Why no reference-point shift is needed
   * - ``prismatic``
     - One linear-axis scalar.
     - ``public[i]`` is copied to ``sap[i]``.
     - The value is a speed or force along one declared axis, not a 6D
       rigid-body twist.
   * - ``revolute``
     - One angular-axis scalar.
     - ``public[i]`` is copied to ``sap[i]``.
     - The value is an angular speed or torque about one declared axis, with no
       linear component to move between COM and body origin.
   * - ``ball``
     - Three angular scalars.
     - The three entries are copied in their stored order.
     - The segment contains only angular velocity or torque components. There
       is no paired linear velocity or force component whose reference point can
       change.
   * - ``fixed``
     - Empty.
     - Nothing is copied.
     - The joint has no velocity or force DOFs.
   * - ``D6`` / general multi-axis
     - Linear-axis scalars first, then angular-axis scalars, matching
       ``joint_dof_dim`` and ``joint_axis``.
     - Every entry is copied at the same offset; linear entries are not moved
       after angular entries.
     - Each scalar is tied to its declared axis, and the motion subspace maps
       that scalar into the internal spatial-vector layout.

The angular-first SAP convention applies after the solver has assembled an
actual spatial vector. It does not reorder these packed generalized-coordinate
segments at the public boundary.

Changed DOF Segments
~~~~~~~~~~~~~~~~~~~~

Free and distance joints do change at the public/SAP boundary. Their six
velocity or generalized-force entries form a complete rigid-body twist or
wrench, so conversion must both reorder the angular and linear blocks and shift
the linear velocity or moment between the center of mass and body origin.

Notation
^^^^^^^^

For one child body, let:

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`W`
     - World frame. All vector components in this page are expressed in
       :math:`W`.
   * - :math:`O`
     - Child body origin, i.e. the body frame origin stored by runtime body
       poses.
   * - :math:`C`
     - Child body center of mass.
   * - :math:`r_{OC}^W`
     - Vector from :math:`O` to :math:`C`, expressed in world coordinates.
   * - :math:`v_O^W`, :math:`v_C^W`
     - Linear velocity of the body origin and center of mass.
   * - :math:`\omega^W`
     - Angular velocity.
   * - :math:`f^W`
     - Applied force.
   * - :math:`\tau_O^W`, :math:`\tau_C^W`
     - Moment measured about :math:`O` or :math:`C`.

Define the skew operator :math:`[a]_\times` by

.. math::

   [a]_\times b = a \times b,
   \qquad
   [a]_\times
   =
   \begin{bmatrix}
   0 & -a_z & a_y \\
   a_z & 0 & -a_x \\
   -a_y & a_x & 0
   \end{bmatrix}.

The only geometric operation in the convention conversion is a reference-point
shift. It is not a frame rotation. Because every component is already expressed
in :math:`W`, changing from :math:`C` to :math:`O` only adds cross-product terms.

Velocity Reference-Point Shift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rigid-body linear velocity at two points on the same body satisfies

.. math::

   v_C^W = v_O^W + \omega^W \times r_{OC}^W.

Equivalently,

.. math::

   v_O^W
   =
   v_C^W - \omega^W \times r_{OC}^W
   =
   v_C^W + [r_{OC}^W]_\times \omega^W.

The public free-joint velocity is stored with linear COM velocity first:

.. math::

   u_{\mathrm{public}}
   =
   \begin{bmatrix}
   v_C^W \\
   \omega^W
   \end{bmatrix}.

The SAP free-joint velocity is stored with angular velocity first, and its
linear component is measured at the body origin:

.. math::

   u_{\mathrm{sap}}
   =
   \begin{bmatrix}
   \omega^W \\
   v_O^W
   \end{bmatrix}.

Therefore

.. math::

   u_{\mathrm{sap}}
   =
   P_v(r_{OC}^W) u_{\mathrm{public}},
   \qquad
   P_v(r)
   =
   \begin{bmatrix}
   0 & I \\
   I & [r]_\times
   \end{bmatrix}.

The inverse conversion used when writing a public ``state_out`` is

.. math::

   u_{\mathrm{public}}
   =
   P_v(r_{OC}^W)^{-1} u_{\mathrm{sap}},
   \qquad
   P_v(r)^{-1}
   =
   \begin{bmatrix}
   -[r]_\times & I \\
   I & 0
   \end{bmatrix},

or, in vector form,

.. math::

   \begin{bmatrix}
   v_C^W \\
   \omega^W
   \end{bmatrix}
   =
   \begin{bmatrix}
   v_O^W + \omega^W \times r_{OC}^W \\
   \omega^W
   \end{bmatrix}.

Force and Wrench Duality
^^^^^^^^^^^^^^^^^^^^^^^^

Forces and velocities must transform as dual coordinates: the instantaneous
power must not depend on whether the twist is represented at :math:`C` or
:math:`O`. The public wrench is

.. math::

   g_{\mathrm{public}}
   =
   \begin{bmatrix}
   f^W \\
   \tau_C^W
   \end{bmatrix},

while the SAP generalized force stores moment first, measured about the body
origin:

.. math::

   g_{\mathrm{sap}}
   =
   \begin{bmatrix}
   \tau_O^W \\
   f^W
   \end{bmatrix}.

The moment shift is

.. math::

   \tau_O^W
   =
   \tau_C^W + r_{OC}^W \times f^W
   =
   \tau_C^W + [r_{OC}^W]_\times f^W.

Thus

.. math::

   g_{\mathrm{sap}}
   =
   P_f(r_{OC}^W) g_{\mathrm{public}},
   \qquad
   P_f(r)
   =
   \begin{bmatrix}
   [r]_\times & I \\
   I & 0
   \end{bmatrix}.

This force transform is the inverse transpose of the velocity transform:

.. math::

   P_f(r) = P_v(r)^{-T}.

That identity gives the power check:

.. math::

   g_{\mathrm{public}}^T u_{\mathrm{public}}
   =
   f^W \cdot v_C^W + \tau_C^W \cdot \omega^W
   =
   \tau_O^W \cdot \omega^W + f^W \cdot v_O^W
   =
   g_{\mathrm{sap}}^T u_{\mathrm{sap}}.

This is the reason SAP Warp shifts both velocity and force. Changing only the
array order would produce the right shape but the wrong generalized work.

Free and Distance Joint Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For free and distance joints, the six velocity entries are:

.. math::

   \mathrm{public}
   =
   \begin{bmatrix}
   v_C^W \\
   \omega^W
   \end{bmatrix},
   \qquad
   \mathrm{sap}
   =
   \begin{bmatrix}
   \omega^W \\
   v_C^W - \omega^W \times r_{OC}^W
   \end{bmatrix}.

The corresponding force entries are:

.. math::

   \mathrm{public}
   =
   \begin{bmatrix}
   f^W \\
   \tau_C^W
   \end{bmatrix},
   \qquad
   \mathrm{sap}
   =
   \begin{bmatrix}
   \tau_C^W + r_{OC}^W \times f^W \\
   f^W
   \end{bmatrix}.

For all other joint types, SAP Warp treats the joint-space arrays as already in
the model's stored order:

.. math::

   u_{\mathrm{sap}} = u_{\mathrm{public}},
   \qquad
   g_{\mathrm{sap}} = g_{\mathrm{public}}.

External Body Forces
^^^^^^^^^^^^^^^^^^^^

``SapState.body_f`` uses the same physical public wrench convention as
free-joint generalized forces:

.. math::

   F_{\mathrm{body,public}}
   =
   \begin{bmatrix}
   f^W \\
   \tau_C^W
   \end{bmatrix}.

The free-motion kernels assemble an inverse-dynamics residual. In that residual
the applied external wrench enters with a negative sign, and the internal body
force buffer is angular-first and body-origin:

.. math::

   F_{\mathrm{body,sap}}
   =
   -P_f(r_{OC}^W)
   F_{\mathrm{body,public}}
   =
   \begin{bmatrix}
   -(\tau_C^W + r_{OC}^W \times f^W) \\
   -f^W
   \end{bmatrix}.

The negative sign is not a user-facing sign convention. It is an internal
residual convention used by the free-motion solve. Callers should normally keep
``SapState.body_f`` in public order and let
:meth:`~sim.solver_sap.SolverSAP.step` build the internal buffer.

Boundary Flags
~~~~~~~~~~~~~~

The boundary dataclasses store explicit flags so the solver knows whether a
buffer is already SAP-native or should be converted:

Pose and generalized-position buffers do not have a ``public``/``sap`` order
flag. Their storage is fixed; only velocity and force-like buffers need the
boundary convention below.

.. list-table::
   :header-rows: 1

   * - Buffer
     - Flag
     - ``"public"`` means
     - ``"sap"`` means
   * - :class:`~sim.sap_runtime.SapState` ``joint_qd``
     - ``joint_qd_order``
     - Free/distance joints store :math:`[v_C^W,\omega^W]`.
     - Free/distance joints store :math:`[\omega^W,v_O^W]`.
   * - :class:`~sim.sap_runtime.SapControl` ``joint_f``
     - ``joint_f_order``
     - Free/distance joints store :math:`[f^W,\tau_C^W]`.
     - Free/distance joints store :math:`[\tau_O^W,f^W]`.
   * - :class:`~sim.sap_runtime.SapState` ``body_f``
     - ``body_f_order``
     - Body wrenches store :math:`[f^W,\tau_C^W]`.
     - Body wrenches are already internal angular-first residual forces.

:meth:`~sim.sap_runtime.SapModel.state` and
:meth:`~sim.sap_runtime.SapModel.control` return public-order buffers. The raw
dataclass defaults are ``"sap"`` because internal kernels also construct
temporary dataclass views after conversion.

Solver Boundary Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~

At a timestep boundary, :meth:`~sim.solver_sap.SolverSAP.step` follows this
sequence:

1. If ``state_in.joint_qd_order == "public"``, convert free and distance joint
   velocities with :math:`P_v(r)`.
2. If ``control.joint_f_order == "public"``, convert free and distance joint
   forces with :math:`P_f(r)`.
3. If ``state_in.body_f_order == "public"``, convert external body forces with
   :math:`-P_f(r)`.
4. Run free motion, contact Jacobian assembly, and contact solve in SAP order.
5. If ``state_out.joint_qd_order == "public"``, convert solved velocities back
   with :math:`P_v(r)^{-1}`.

The conversion is repeated each step because :math:`r_{OC}^W` depends on the
current body orientation:

.. math::

   r_{OC}^W = R_{WB}\,r_{OC}^B.

Practical Rule
--------------

For application code, use :meth:`~sim.sap_runtime.SapModel.state` and
:meth:`~sim.sap_runtime.SapModel.control`, leave ``joint_qd_order``,
``joint_f_order``, and ``body_f_order`` as ``"public"``, and let
:meth:`~sim.solver_sap.SolverSAP.step` perform the boundary conversion. Use
``"sap"`` only when an array has already been converted into angular-first,
body-origin form and you want to bypass the public boundary adapter.
