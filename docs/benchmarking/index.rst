Benchmarking
============

The primary simulation entry point is ``benchmark.py``. It loads a scene file,
constructs SAP runtime data, creates a collision pipeline, and steps
:class:`~sim.solver_sap.SolverSAP` for a fixed duration.

Quick Run
---------

Run the default G1 USD scene. The scene file defaults to 1024 worlds, so the
small smoke-test form overrides the world count and frame count:

.. code-block:: shell

   uv run python benchmark.py --frames 2 --num-worlds 1

Run a specific YAML scene:

.. code-block:: shell

   uv run python benchmark.py \
     --scene assets/yaml/unitree_h1_usd.yaml \
     --duration 1.0 \
     --device cuda:0

Useful Runtime Flags
--------------------

``--scene``
   YAML or JSON scene file. Defaults to
   ``assets/yaml/unitree_g1_usd.yaml``.

``--duration``
   Simulated time in seconds. Ignored when ``--frames`` is supplied.

``--frames``
   Exact number of solver steps. This is useful for smoke tests because it
   avoids reasoning about ``ceil(duration / dt)``.

``--dt``
   Timestep. Defaults to ``simulation.dt`` in the scene file, then ``0.003``.

``--num-worlds``
   Number of replicated worlds. Defaults to ``simulation.num_worlds``.

``--device``
   Warp device string, for example ``cpu`` or ``cuda:0``.

Benchmark Output
----------------

During the loop, ``benchmark.py`` prints:

.. code-block:: text

   frame 1 sim_time 0.003
   frame 2 sim_time 0.006

The final summary contains:

``scene``
   Resolved scene path.

``device``
   Warp device used for allocation and kernels.

``dt`` and ``frames``
   Effective timestep and number of solver steps.

``num_worlds``
   Number of independent replicated worlds after command-line overrides.

``max_rigid_contact_per_env``
   Per-world contact cap passed to :class:`~sim.solver_sap.SolverSAP`.

``rigid_contact_capacity``
   Flat contact buffer capacity passed to
   :class:`~sim.collision.pipeline.SapCollisionPipeline`.

``cuda_graph``
   Whether the benchmark captured and launched the native step as a CUDA graph.

``elapsed``, ``fps``, and ``realtime_ratio``
   Wall-clock timing, simulated frames per second, and simulated seconds per
   wall-clock second.

Benchmark Loop
--------------

The benchmark uses the scene configuration to choose ``dt``, ``num_worlds``,
per-env ``max_rigid_contact``, and :class:`~sim.solver_sap.SolverSAP` keyword
arguments. The solver keeps per-environment contact slots, while
:class:`~sim.collision.pipeline.SapCollisionPipeline` writes into one flat
contact buffer sized for all worlds:

.. code-block:: python

   max_rigid_contact_per_env = simulation["max_rigid_contact"]
   rigid_contact_capacity = max_rigid_contact_per_env * num_worlds

   loaded = load_sap_scene(scene_path, device=device, rigid_contact_max=rigid_contact_capacity)
   solver = SolverSAP(loaded.sap_model, max_rigid_contact=max_rigid_contact_per_env, **solver_kwargs)
   collision_pipeline = SapCollisionPipeline(loaded.collision_model, rigid_contact_max=rigid_contact_capacity)
   contacts = collision_pipeline.contacts()

   steps = int(duration / dt)
   for _ in range(steps):
       state_0.clear_forces()
       collision_pipeline.collide(sap_collision_state_from_state(state_0), contacts)
       solver.step(state_0, state_1, control, contacts, dt)
       state_0, state_1 = state_1, state_0

On CUDA devices, the benchmark attempts to capture the native step as a CUDA
graph. If capture fails or the device is not CUDA-capable, it falls back to the
regular Python loop.

Choosing Scene Size
-------------------

Use ``--num-worlds 1`` for correctness debugging and documentation examples.
Increase ``--num-worlds`` only after a single world is stable. Because the flat
collision capacity is ``max_rigid_contact * num_worlds``, large batches can use
substantial memory even when each environment has a moderate contact cap.

Use ``simulation.max_rigid_contact`` to size the per-world solver buffers. If
contacts are truncated, increase that value, reduce ``shape_gap``/``rigid_gap``
where appropriate, or simplify collision geometry.
