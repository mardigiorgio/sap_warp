Collision
=========

SAP Warp treats collision generation as a separate runtime stage. The collision
pipeline reads body poses and shape metadata, writes a bounded
:class:`~sim.collision.contacts.SapContacts` buffer, and leaves contact response
to :class:`~sim.solver_sap.SolverSAP`.

This collision front end is based on
`Newton <https://github.com/newton-physics/newton>`__'s collision code,
including its broad-phase/narrow-phase organization and shape metadata. SAP
Warp adapts that code path to emit SAP-owned rigid contact buffers while
preserving compatibility with Newton-authored assets and runtime behavior.

Hydroelastic contact support is in development; the documented runtime path on
this page is rigid contact generation into
:class:`~sim.collision.contacts.SapContacts`.

Pipeline Stages
---------------

:meth:`~sim.collision.pipeline.SapCollisionPipeline.collide` runs these stages:

1. Clear the output contact buffer.
2. Compute world-space shape AABBs from body poses, shape transforms, margins,
   and gaps.
3. Run the selected broad phase to produce candidate shape pairs.
4. Run narrow phase algorithms for candidate pairs.
5. Write rigid contacts into :class:`~sim.collision.contacts.SapContacts`
   until ``rigid_contact_max`` is reached.

The solver consumes contact shape ids, body-frame contact points, normals,
margins, and optional material attributes. It does not query meshes or SDFs
directly.

Broad Phase Modes
-----------------

:class:`~sim.collision.pipeline.SapCollisionPipeline` supports three broad
phase modes:

``explicit``
   Uses precomputed shape pairs from the model. This is the default path for
   loaded scenes that provide ``shape_contact_pairs``.

``sap``
   Sweep-and-prune broad phase. Use this when dynamic pair discovery is needed
   and ``model.shape_world`` is available.

``nxn``
   All-pairs broad phase filtered by world and shape flags. This is useful for
   debugging small scenes.

The constructor also accepts expert broad-phase and narrow-phase instances, but
both must be supplied together.

Supported Shape Data
--------------------

The runtime shape type enum includes planes, heightfields, spheres, capsules,
ellipsoids, cylinders, boxes, triangle meshes, cones, and convex meshes. Scene
files expose the common inline authoring names:

.. code-block:: text

   box, sphere, capsule, cylinder, cone, ellipsoid, mesh

USD, URDF, and MJCF import paths map external geometry into the same collision
model arrays.

Margins, Gaps, and Materials
----------------------------

Each shape can carry:

``margin``
   Geometric contact-surface offset. Margin expands the effective collision
   envelope and is subtracted from the signed gap passed to the SAP solve.

``gap``
   Candidate-generation band. Builder ``rigid_gap`` supplies the default when
   a shape does not specify one. Gap widens the range in which the collision
   pipeline emits potential contacts; it does not by itself decide the final
   contact force.

``ke``, ``tau``, ``mu``
   Stiffness, explicitly specified contact dissipation time scale, and friction
   coefficient passed into the SAP contact solve after pairwise combination.
   ``tau`` is material data, not a value derived from ``ke``/``kd``.

``collision_group`` and flags
   Broad-phase and narrow-phase participation controls.

The distinction between ``margin`` and ``gap`` is important for SAP. During
broad phase, each shape AABB is expanded by the sum of its margin and gap:

.. math::

   \mathrm{AABB}_{s,\mathrm{query}}
   =
   \mathrm{AABB}_s \oplus (m_s + g_s).

During narrow phase, a pair of shapes ``a`` and ``b`` can generate a rigid
contact candidate when the margin-adjusted separation is inside the pair gap:

.. math::

   d_{ab} - (m_a + m_b) \le g_a + g_b.

The contact Jacobian stage then stores the margin-adjusted signed gap

.. math::

   \phi_0
   =
   n^T(x_1 - x_0) - m_a - m_b,

where :math:`x_0` and :math:`x_1` are the witness points and :math:`n` is the
contact normal. A larger margin changes :math:`\phi_0` and therefore changes
the effective contact surface seen by the solver.

The gap term instead acts as an anticipation band. It allows the collision
pipeline to hand SAP a contact candidate even when :math:`\phi_0` is still
slightly positive:

.. math::

   0 < \phi_0 \le g_a + g_b.

SAP then decides the contact impulse/force inside the velocity-level objective;
collision detection only supplies candidates and material data. This is useful
for stability because the solver can see near-future contacts before visible
penetration occurs, reducing contact popping and missed contacts at larger
timesteps or higher relative velocities. The tradeoff is capacity: increasing
``gap`` may increase the number of candidates, so ``rigid_contact_max`` must be
large enough for the broader band.

Capacity
--------

``rigid_contact_max`` bounds the flat contact buffer owned by the collision
pipeline. Scene files set ``simulation.max_rigid_contact`` as a per-world cap
for benchmark runs; with replicated worlds, ``benchmark.py`` passes
``simulation.max_rigid_contact * num_worlds`` to
:class:`~sim.collision.pipeline.SapCollisionPipeline`. When generated contacts
exceed capacity, extra contacts are dropped and the solver exposes the most
recent truncated count through
``solver.last_truncated_contact_count``.

Common Pattern
--------------

.. code-block:: python

   collision = SapCollisionPipeline(scene.collision_model, rigid_contact_max=256)
   contacts = collision.contacts()

   state_0.clear_forces()
   collision.collide(sap_collision_state_from_state(state_0), contacts)
   solver.step(state_0, state_1, control, contacts, dt)

Run collision once per solver step unless the scene is known to have fixed
contact topology and the caller intentionally reuses a contact set.
