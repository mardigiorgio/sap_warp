Loader
======

Scene files are YAML or JSON documents consumed by
:func:`~sim.loader.scene.load_sap_scene`. They describe simulation timing,
contact capacity, solver kwargs, builder defaults, external assets, optional
inline bodies, articulated joints, initial state edits, and deterministic
post-build edits.

The current schema version is ``1``.

Load Order
----------

The loader applies a scene in this order:

1. Read ``simulation.num_worlds`` or ``replicate.num_worlds`` and optional
   replication spacing.
2. Apply supported ``builder`` defaults.
3. Import external ``assets``.
4. Add inline ``bodies`` and their shapes.
5. Add inline ``joints`` and ``articulations``.
6. Apply ``initial_joint_q``.
7. Apply ``post_build`` operations.
8. Replicate the authored world.
9. Add the optional world ``ground`` plane.

This order matters. For example, ``post_build`` edits run before replication,
so a joint-target edit is copied into every replicated world. The ground plane
is added after replication and remains a shared world shape.

Top-Level Sections
------------------

``schema_version``
   Integer schema marker. The current loader supports ``1``.

``name``
   Human-readable scene identifier.

``simulation``
   Runtime settings used by ``benchmark.py``. Common keys are ``dt``,
   ``num_worlds``, ``max_rigid_contact``, and ``solver``.

``builder``
   Scene builder defaults. The current loader consumes ``rigid_gap``,
   ``default_body_armature``, and ``defaults.joint``/``defaults.shape`` from
   this block. Other builder attributes can be changed with
   ``post_build: [{op: set_attr, ...}]`` when supported.

``ground``
   Optional world plane. It is enabled by default unless set to ``false`` or
   ``enabled: false``.

``assets``
   External ``usd``, ``urdf``, or ``mjcf`` assets. Sources may be local paths,
   inline URDF XML strings, or git-backed sparse checkouts.

``bodies``, ``joints``, and ``articulations``
   Inline procedural bodies and generalized-coordinate structure.

``initial_joint_q``
   Direct initial generalized position edits by index or joint reference.

``post_build``
   Deterministic edits after assets and inline objects are added.

``replicate``
   Optional world replication settings. ``simulation.num_worlds`` is also
   accepted and is what the current example scenes use.

Minimal Inline Scene
--------------------

This is the smallest useful shape of a procedural scene: one free box, one
ground plane, and one solver configuration.

.. code-block:: yaml

   schema_version: 1
   name: single_box

   simulation:
     dt: 0.003
     num_worlds: 1
     max_rigid_contact: 64
     solver:
       contact_preset_variant: approx32
       line_search_variant: monotone_decay

   builder:
     rigid_gap: 0.002
     defaults:
       shape:
         density: 1000.0
         ke: 1000000.0
         tau: 0.01
         mu: 0.5

   ground:
     enabled: true
     label: DEFAULT_GROUND

   bodies:
     - id: box
       mass: 1.0
       transform:
         p: [0.0, 0.0, 1.0]
         q: identity
       shapes:
         - type: box
           hx: 0.5
           hy: 0.5
           hz: 0.5

   joints:
     - id: box_free
       type: free
       parent: world
       child: box

   articulations:
     - id: box_articulation
       joints: [box_free]

Run it with:

.. code-block:: shell

   uv run python benchmark.py --scene path/to/single_box.yaml --frames 10

Simulation Block
----------------

``benchmark.py`` reads:

.. code-block:: yaml

   simulation:
     dt: 0.003
     num_worlds: 1
     max_rigid_contact: 48
     solver:
       contact_preset_variant: drake
       line_search_variant: armijo_decay

``dt``
   Default timestep. The command-line ``--dt`` flag overrides it.

``num_worlds``
   Number of independent replicated worlds. The command-line ``--num-worlds``
   flag overrides it.

``max_rigid_contact``
   Per-world solver contact cap. With replicated worlds, ``benchmark.py``
   allocates the flat collision contact buffer as
   ``max_rigid_contact * num_worlds``.

``solver``
   Keyword arguments passed to :class:`~sim.solver_sap.SolverSAP`. See
   :doc:`solver` for supported solver options.

Builder Defaults
----------------

``rigid_gap``
   Default shape ``gap`` used when a shape does not provide one. This is a
   contact candidate generation band, not a force parameter.

``default_body_armature``
   Default rigid-body armature for bodies created without an explicit
   per-body armature. The value is added to the diagonal of ``body_inertia``;
   it is body-space rotational inertia regularization, not joint DOF
   armature. Values that are reasonable for robot links can dominate tiny
   free objects, so keep this near zero when small bodies should spin down
   from contact using their physical inertia.

``defaults.shape``
   Base material and collision metadata for subsequently created shapes.

``defaults.joint``
   Base drive, limit, armature, effort, velocity, and friction metadata for
   subsequently created joint DOFs. Joint armature is stored in
   ``joint_armature`` and is separate from ``default_body_armature``.

Numeric values may be numbers or simple expressions containing ``pi`` and
``+``, ``-``, ``*``, or ``/``:

.. code-block:: yaml

   angle: pi * 0.5

Transforms
----------

Transforms appear on assets, bodies, shapes, and joint frames.

``identity``
   Identity transform.

``p``
   Translation vector ``[x, y, z]``.

``q``
   Quaternion as ``identity``, ``[x, y, z, w]``, or an ``axis_angle`` mapping.

Example:

.. code-block:: yaml

   xform:
     p: [0.0, 0.0, 0.62]
     q:
       axis_angle:
         axis: [0.0, 0.0, 1.0]
         angle: pi * 0.5

Asset Sources
-------------

A local source is a path string or mapping with ``path``:

.. code-block:: yaml

   assets:
     - id: robot
       type: usd
       source: assets/robots/robot.usda

Git sources fetch sparse checkouts into the SAP asset cache:

.. code-block:: yaml

   assets:
     - id: g1
       type: usd
       source:
         git:
           repo: https://github.com/newton-physics/newton-assets.git
           rev: 261cd1f429619d8ef4f546bd788ab9dea906b5e1
           sparse:
             - unitree_g1/usd_structured
             - unitree_g1/meshes
           path: unitree_g1/usd_structured/g1_29dof_with_hand_rev_1_0.usda

Git-backed assets are cached under ``SAP_WARP_ASSET_CACHE`` when set, otherwise
under ``~/.cache/sap_warp/assets``. ``SAP_WARP_ASSET_OFFLINE=1`` disables
fetching and requires the cache entry to already exist.

Imported Asset Options
----------------------

Common options accepted by imported assets include:

``xform``
   Root transform with ``p`` and ``q`` fields.

``cfg``, ``shape``, or ``physics``
   Shape/material overrides applied while importing the asset. ``cfg`` may
   contain any shape config field; ``shape`` and ``physics`` are normalized for
   common material aliases such as ``ke``, ``tau``, ``mu``, ``margin``, and
   ``gap``.

``enable_self_collisions``
   Whether shapes from the same imported asset may collide with one another.

``hide_collision_shapes`` and ``force_show_colliders``
   Visibility flags stored in shape flags. They do not disable collision.

``skip_mesh_approximation``
   For USD import, keep mesh shapes as meshes when possible instead of applying
   mesh approximation.

``ignore_paths``
   USD path patterns to skip.

``floating``
   URDF/MJCF option that creates a floating base when supported by that
   importer path.

``load_visual_shapes`` and ``parse_visuals_as_colliders``
   URDF/MJCF options controlling whether visual geometry is imported and
   whether it participates in collision.

The current imported-asset path assumes z-up assets. Unsupported asset features
are collected in ``SapLoadedScene.unsupported_features``; with ``strict=True``
they raise :class:`~sim.loader.scene.SapUnsupportedSceneFeature`.

Inline Bodies and Shapes
------------------------

Inline body fields:

``id``
   Stable identifier used by inline joints and post-build body references.

``label``
   Optional runtime body label. Defaults to ``id``.

``transform``
   Initial body pose.

``mass``
   Initial body mass. Shape density can add mass and inertia unless inertia is
   locked by an imported asset path.

``shape``/``physics`` fields or direct material aliases
   Body-level defaults for child shapes.

Inline shapes support ``box``, ``sphere``, ``capsule``, ``cylinder``, ``cone``,
``ellipsoid``, and ``mesh``:

.. list-table::
   :header-rows: 1

   * - Shape
     - Size fields
   * - ``box``
     - ``hx``, ``hy``, ``hz`` half-extents.
   * - ``sphere``
     - ``radius``.
   * - ``capsule``, ``cylinder``, ``cone``
     - ``radius`` and ``half_height``.
   * - ``ellipsoid``
     - ``a``, ``b``, ``c`` radii.
   * - ``mesh``
     - Mesh geometry fields accepted by the loader.

Shape-level ``cfg`` overrides body and builder defaults:

.. code-block:: yaml

   bodies:
     - id: payload
       transform:
         p: [0.0, 0.0, 1.0]
         q: identity
       shapes:
         - type: box
           hx: 0.2
           hy: 0.2
           hz: 0.2
           cfg:
             density: 500.0
             mu: 0.8
             gap: 0.004

Inline Joints and Articulations
-------------------------------

Inline joints support ``fixed``, ``free``, ``revolute``, and ``prismatic``.

``parent`` and ``child``
   Body ids or ``world`` for the parent.

``parent_xform`` and ``child_xform``
   Joint frames expressed in parent and child body frames.

``axis``
   Revolute or prismatic axis. Use ``[x, y, z]``, ``x``/``y``/``z``-style
   tokens accepted by the loader, or the integer axis tokens used internally.

``collision_filter_parent``
   Defaults to ``true``. When enabled, directly connected parent and child
   shapes are filtered from collision.

``articulations``
   Lists joint ids in each articulation. The solver requires at least one joint
   DOF; a scene with only fixed joints is not enough for
   :class:`~sim.solver_sap.SolverSAP`.

Example:

.. code-block:: yaml

   joints:
     - id: hinge
       type: revolute
       parent: world
       child: link
       axis: [0.0, 0.0, 1.0]
       limit_lower: -1.57
       limit_upper: 1.57

   articulations:
     - id: hinge_articulation
       joints: [hinge]

Initial Joint Positions
-----------------------

``initial_joint_q`` edits entries in the generalized position array before the
initial forward-kinematics pass:

.. code-block:: yaml

   initial_joint_q:
     - joint: revolute_a_b
       value: pi * 0.5
     - joint: box_free
       offset: 2
       value: 1.25

Use ``joint`` or ``joint_label`` to select a joint, plus an optional ``offset``
inside that joint's ``q`` block. Use ``index`` to edit an absolute
``joint_q`` index.

Shape Defaults
--------------

Shape defaults and per-shape ``cfg`` blocks use the same fields. Common fields
are:

``density``
   Density used when the loader computes body mass and inertia from shapes.

``ke`` and ``tau``
   Contact stiffness and explicitly specified contact dissipation time scale.
   ``tau`` is read as material data; it is not computed from ``ke`` or a
   damping coefficient.

``mu``
   Coulomb friction coefficient.

``margin`` and ``gap``
   ``margin`` offsets the effective contact surface and enters the signed gap
   used by the SAP solve. ``gap`` is an anticipation band for generating
   potential contact candidates before penetration. A pair can be emitted when
   its margin-adjusted separation satisfies

   .. math::

      d_{ab} - (m_a + m_b) \le g_a + g_b.

   Contacts with small positive :math:`\phi_0` are then passed to the solver,
   which decides the force/impulse through the SAP objective. Increasing
   ``gap`` can improve contact stability, but it can also increase contact
   count, so keep ``simulation.max_rigid_contact`` sized accordingly. ``gap``
   defaults to the builder ``rigid_gap``.

``collision_group``
   Broad-phase group mask. Group ``0`` does not collide. Positive groups collide
   with matching positive groups and with negative groups according to the
   loader's group-pair test.

``has_shape_collision`` and ``has_particle_collision``
   Collision participation flags.

``is_hydroelastic``
   Marks a shape for hydroelastic contact support, which is in development.

``is_visible`` and ``is_site``
   Visibility and site flags. Sites are non-colliding, zero-density shapes.

Useful aliases are normalized before application:

.. list-table::
   :header-rows: 1

   * - Canonical field
     - Accepted aliases
   * - ``margin``
     - ``contact_margin``, ``shape_margin``, ``margin``
   * - ``gap``
     - ``shape_gap``, ``gap``
   * - ``mu``
     - ``mu``, ``shape_mu``
   * - ``ke``
     - ``ke``, ``shape_ke``
   * - ``tau``
     - ``tau``, ``shape_tau``, ``relaxation_time``

Joint Defaults
--------------

Joint defaults and joint definitions can set:

``axis``
   Local joint axis.

``target_pos``, ``target_vel``, ``target_ke``, ``target_kd``
   Joint drive target and gains.

``limit_lower``, ``limit_upper``, ``limit_ke``, ``limit_kd``
   Joint limit bounds and stiffness/damping.

``armature``, ``effort_limit``, ``velocity_limit``, ``friction``
   Per-DOF solver parameters. ``armature`` adds diagonal generalized inertia
   on the joint DOF through ``joint_armature``; it does not modify
   ``body_inertia``. Use ``builder.default_body_armature`` for rigid-body
   inertia regularization.

``actuator_mode``
   Integer actuator mode stored on the runtime joint DOF.

Post-Build Operations
---------------------

Supported ``post_build`` operations include:

``add_shape``
   Add another inline shape to a body or the world.

``set_array``
   Edit a supported builder array with a scalar ``value``, explicit ``values``,
   or values copied ``from_array``.

``copy_array``
   Copy entries between supported builder arrays.

``scale_sphere_shapes``
   Multiply every sphere radius by ``factor``.

``approximate_meshes``
   Replace colliding mesh shapes with oriented bounding boxes when
   ``method: bounding_box`` is selected.

``set_attr``
   Set a supported builder attribute, for example ``gravity``.

``set_joint_targets``
   Set target mode and gains by index range, exact joint label, label prefix,
   or substring match.

``set_joint_q``
   Set one generalized position entry.

``copy_joint_q_to_joint_targets``
   Seed joint position targets from current ``joint_q``.

``set_joint_armature``
   Assign armature values over a selected range.

Supported array names for ``set_array`` and ``copy_array`` are:

.. code-block:: text

   shape_margin, shape_gap, shape_material_ke, shape_material_tau,
   shape_material_mu, shape_material_restitution,
   shape_material_mu_torsional, shape_material_mu_rolling,
   shape_material_kh, shape_collision_group, shape_flags,
   joint_q, joint_qd, joint_target_pos, joint_target_vel,
   joint_target_ke, joint_target_kd, joint_target_mode, joint_armature

Selectors and Ranges
--------------------

Array operations accept:

``index``
   Single index. Negative indices are relative to the end.

``range: all``
   Every entry.

``range: head:N``
   First ``N`` entries.

``range: tail:N``
   Last ``N`` entries.

``range: from:N``
   Entries from ``N`` to the end.

``range: [start, end]``
   Half-open range. ``end: null`` means the end of the array.

Joint-selector operations accept:

``joint`` or ``joint_label``
   Exact joint label, or a suffix match after ``/`` when the imported asset
   prefixes labels.

``joints`` or ``joint_labels``
   List of joint references.

``label_prefix``
   Every joint label that starts with the prefix.

``match``
   Every joint label containing the substring.

Examples:

.. code-block:: yaml

   post_build:
     - op: set_array
       array: joint_q
       range: [0, 3]
       values: [0.0, 0.0, 0.67]

     - op: set_joint_targets
       range: from:6
       mode: position
       ke: 500.0
       kd: 10.0

     - op: set_joint_targets
       label_prefix: robot/left_leg
       mode: position
       ke: 1000.0
       kd: 20.0

Replication
-----------

Replicated worlds are independent copies of authored bodies, shapes, joints,
articulations, controls, and labels. The loader offsets each world by
``replicate.spacing``. With two non-zero spacing axes, worlds are arranged in a
centered grid; with one non-zero axis, they are arranged in a line.

.. code-block:: yaml

   simulation:
     num_worlds: 16

   replicate:
     spacing: [2.0, 2.0, 0.0]

``load_sap_scene(..., num_worlds=...)`` and the benchmark ``--num-worlds`` flag
override the scene value.

Existing Examples
-----------------

``assets/yaml/unitree_g1_usd.yaml``
   G1 USD scene, Drake preset, Armijo line search, high default world count for
   throughput, and mesh-to-box post-build approximation.

``assets/yaml/unitree_h1_usd.yaml``
   H1 USD scene, Drake preset, Armijo line search, and full-position joint
   targets.

``assets/yaml/anymal_c_urdf.yaml``
   ANYmal C URDF scene with a floating base, sphere scaling, joint target
   setup, and per-DOF armature edits.

``assets/yaml/anymal_d_usd.yaml``
   ANYmal D USD scene with explicit initial base ``joint_q`` edits and target
   setup.

``assets/yaml/multi_joints.yaml``
   Small inline scene covering fixed, revolute, prismatic, and free joints.

Runtime Data
------------

SAP Warp separates authoring-time scene data from runtime arrays. Scene files
describe assets, inline bodies, defaults, solver options, and post-build edits.
The loader turns that description into compact Warp arrays consumed by the
collision pipeline and solver. The runtime model, state, control, and contact
data structures produced by the loader are based on
`Newton <https://github.com/newton-physics/newton>`__'s runtime code. SAP Warp
adapts those structures so it can wrap Newton-owned Warp arrays directly where
possible, preserve Newton-style ordering metadata at the public boundary, and
keep imported USD/URDF/MJCF scenes close to Newton behavior.

Load Result
~~~~~~~~~~~

:func:`~sim.loader.scene.load_sap_scene` returns
:class:`~sim.loader.scene.SapLoadedScene` with:

``collision_model``
   Shape metadata for broad phase, narrow phase, and contact writing.

``collision_state``
   Initial collision body poses.

``sap_model``
   Articulated solver model.

``sap_state``
   Initial generalized and body state.

``sap_control``
   Initial joint forces, targets, and actuator arrays.

``body_labels`` and ``shape_labels``
   Stable labels useful for scene edits, diagnostics, and debugging.

State and Control
~~~~~~~~~~~~~~~~~

:meth:`~sim.sap_runtime.SapModel.state` clones the model's initial state into a
mutable :class:`~sim.sap_runtime.SapState`.
:meth:`~sim.sap_runtime.SapModel.control` clones control arrays into
:class:`~sim.sap_runtime.SapControl`. The benchmark uses two states and swaps
them after every step:

.. code-block:: python

   state_0 = loaded.sap_state
   state_1 = loaded.sap_model.state()
   control = loaded.sap_control

   solver.step(state_0, state_1, control, contacts, dt)
   state_0, state_1 = state_1, state_0

:meth:`~sim.sap_runtime.SapState.clear_forces` clears external body forces.
:meth:`~sim.sap_runtime.SapControl.clear` clears direct joint forces and target
arrays.

Ordering Conventions
~~~~~~~~~~~~~~~~~~~~

The public runtime state uses the model-facing order, while the SAP kernels use
an angular-first, body-origin order for free and distance joints. The full
reference-point shifts, formulas, and order flags are documented in
:doc:`convention`.

World Replication
~~~~~~~~~~~~~~~~~

Scene files may replicate one authored world into many independent worlds with
``simulation.num_worlds`` or a ``replicate`` block. Replication duplicates
bodies, shapes, joints, controls, and labels while preserving shared global
objects such as the ground plane when applicable.

The G1 benchmark scene uses many worlds by default for throughput measurement.
Override ``--num-worlds`` during smoke tests.

Asset Resolution
~~~~~~~~~~~~~~~~

Asset sources may be local paths or git-backed sparse checkouts. Git assets are
cached under ``SAP_WARP_ASSET_CACHE`` or ``~/.cache/sap_warp/assets``. Set
``SAP_WARP_ASSET_OFFLINE=1`` only after the needed assets exist in the cache.

Unsupported Features
~~~~~~~~~~~~~~~~~~~~

The loader records recognized-but-unsupported scene features in
``unsupported_features`` on :class:`~sim.loader.scene.SapLoadedScene`. With
``strict=True``, unsupported features raise
:class:`~sim.loader.scene.SapUnsupportedSceneFeature` instead of silently
continuing.
