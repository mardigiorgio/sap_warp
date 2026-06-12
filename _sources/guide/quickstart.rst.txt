Quick Start
===========

This page walks through the shortest supported path: inspect a scene in the
Viser viewer, then step the same runtime objects directly from Python.

Before You Run
--------------

The default G1 scene is configured for throughput benchmarking with
``simulation.num_worlds: 1024``. For interactive viewing, override it to one
world. The first run may also populate the git-backed asset cache under
``~/.cache/sap_warp/assets`` unless ``SAP_WARP_ASSET_CACHE`` points somewhere
else.

Run a Scene
-----------

Launch the shipped G1 scene in the Viser viewer:

.. code-block:: shell

   uv run python -m viewer.viser.sap_viewer \
     --scene assets/yaml/unitree_g1_usd.yaml \
     --num-worlds 1

The command prints a local URL such as ``http://localhost:8080``. Open that URL
in a browser to inspect the simulation. Leave ``--duration`` and ``--frames``
unset for an interactive run.

For a bounded viewer smoke run, add an exact frame count:

.. code-block:: shell

   uv run python -m viewer.viser.sap_viewer \
     --scene assets/yaml/unitree_g1_usd.yaml \
     --frames 120 \
     --num-worlds 1

Choose Runtime Parameters
-------------------------

The viewer reads defaults from the scene file and lets command-line flags
override the most common runtime values:

.. code-block:: shell

   uv run python -m viewer.viser.sap_viewer \
     --scene assets/yaml/unitree_g1_usd.yaml \
     --num-worlds 1 \
     --dt 0.003 \
     --viewer-fps 60 \
     --device cuda:0

``--duration`` sets simulated time. ``--num-worlds`` overrides
``simulation.num_worlds`` from the scene, and ``--dt`` overrides
``simulation.dt``. ``--frames`` can be used instead of ``--duration`` when you
want an exact step count. ``--viewer-fps`` controls the display update target.
The solver keyword arguments still come from ``simulation.solver`` in the scene
file.

Contact capacity is split into two related numbers:

``simulation.max_rigid_contact``
   Per-world solver contact cap.

``simulation.max_rigid_contact * num_worlds``
   Flat collision buffer capacity used by the viewer and benchmark paths.

If the collision stage produces more contacts than the configured capacity,
extra contacts are dropped and
``solver.last_truncated_contact_count`` reports the most recent truncation
count.

Step from Python
----------------

The minimal Python loop mirrors the benchmark:

.. code-block:: python

   import warp as wp

   from sim.collision.pipeline import SapCollisionPipeline
   from sim.loader.scene import load_sap_scene
   from sim.resources.collision_model import sap_collision_state_from_state
   from sim.solver_sap import SolverSAP

   device = wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")
   num_worlds = 1
   max_rigid_contact_per_env = 48
   rigid_contact_capacity = max_rigid_contact_per_env * num_worlds

   scene = load_sap_scene(
       "assets/yaml/unitree_g1_usd.yaml",
       device=device,
       rigid_contact_max=rigid_contact_capacity,
       num_worlds=num_worlds,
       strict=True,
   )

   solver = SolverSAP(
       scene.sap_model,
       max_rigid_contact=max_rigid_contact_per_env,
       contact_preset_variant="drake",
       line_search_variant="armijo_decay",
   )
   collision = SapCollisionPipeline(
       scene.collision_model,
       rigid_contact_max=rigid_contact_capacity,
   )
   contacts = collision.contacts()

   state_0 = scene.sap_state
   state_1 = scene.sap_model.state()
   control = scene.sap_control
   dt = 0.003

   for _ in range(10):
       state_0.clear_forces()
       collision.collide(sap_collision_state_from_state(state_0), contacts)
       solver.step(state_0, state_1, control, contacts, dt)
       state_0, state_1 = state_1, state_0

Read the loop as a data-flow story:

.. list-table::
   :header-rows: 1

   * - Object
     - Role in the step
   * - ``scene.sap_model``
     - Immutable topology, material, drive, limit, and default state arrays.
   * - ``state_0``
     - Current generalized positions, velocities, body poses, and external
       forces.
   * - ``state_1``
     - Output state written by the solver.
   * - ``control``
     - Joint forces, drive targets, target velocities, and actuation values.
   * - ``contacts``
     - Preallocated contact buffer filled by the collision pipeline.
   * - ``solver``
     - SAP timestepper that consumes state, control, contacts, and ``dt``.

Important details:

* Collision is explicit. Run
  :meth:`~sim.collision.pipeline.SapCollisionPipeline.collide` before each
  solver step when contacts may change.
* :meth:`~sim.solver_sap.SolverSAP.step` writes into ``state_out`` and returns
  that output state.
* State buffers are swapped after each step.
* ``max_rigid_contact`` is the per-world solver cap. The flat collision buffer
  must be large enough for all generated contacts. Dropped contacts are
  reported through ``solver.last_truncated_contact_count``.

Try Another Scene
-----------------

The repository includes both imported-asset and inline-procedural examples:

.. code-block:: shell

   uv run python -m viewer.viser.sap_viewer \
     --scene assets/yaml/multi_joints.yaml \
     --num-worlds 1
   uv run python -m viewer.viser.sap_viewer \
     --scene assets/yaml/unitree_h1_usd.yaml \
     --num-worlds 1
   uv run python -m viewer.viser.sap_viewer \
     --scene assets/yaml/anymal_c_urdf.yaml \
     --num-worlds 1

Use ``--device cpu`` when CUDA is not available. Imported USD/URDF scenes need
the external assets to be fetched or already present in the asset cache.
