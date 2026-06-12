Overview
========

**SAP Warp** is a `Warp`_-based implementation of the SAP contact formulation
from `An Unconstrained Convex Formulation of Compliant Contact`_ with
additional improvements for robust, stable, and efficient simulation of
frictional contact for robotics. The project follows `Drake`_'s implementation
as a reference and exposes a compact Python/Warp runtime for scene loading,
collision generation, solver benchmarking, and visualization.

.. admonition:: Learn More
   :class: tip

   Start with :doc:`quickstart` for a first run. For the solver behavior and
   runtime options, read :doc:`../concepts/solver`.

Core Concepts
-------------

- :func:`~sim.loader.scene.load_sap_scene` turns YAML/JSON scene files,
  imported assets, inline bodies, and post-build edits into runtime arrays.
- :class:`~sim.sap_runtime.SapModel` stores articulated topology, body inertia,
  materials, drives, limits, and shape metadata consumed by the solver.
- :class:`~sim.sap_runtime.SapState` stores generalized positions, generalized
  velocities, body poses, body velocities, and external body forces.
- :class:`~sim.sap_runtime.SapControl` stores direct generalized forces,
  targets, and actuator arrays.
- :class:`~sim.collision.pipeline.SapCollisionPipeline` generates active
  contacts from the collision model and current body poses.
- :class:`~sim.solver_sap.SolverSAP` advances one timestep from ``state_in`` to
  ``state_out``.

The runtime is easiest to read as one loop. Load a scene once, allocate two
states, allocate one control object, allocate one contact buffer, then repeat:
clear forces, refresh contacts from the current poses, solve one SAP timestep,
and swap the state buffers. Most public APIs exist to support one step in that
loop; lower-level Newton-derived kernels are intentionally left out of the main
reference unless they are useful for debugging the SAP path.

Where to Go Next
----------------

* **User Guide** - :doc:`installation` sets up the repository environment, and
  :doc:`quickstart` runs a first scene and steps from Python.
* **Concepts** - :doc:`../concepts/convention`,
  :doc:`../concepts/loader`, :doc:`../concepts/collision`, and
  :doc:`../concepts/solver` explain runtime conventions, scene schema, contact
  generation, and solver behavior.
* **Benchmarking** - :doc:`../benchmarking/index` covers the command-line
  benchmark workflow and runtime sizing.
* **Visualization** - :doc:`../visualization/viser` inspects scenes
  interactively and records MP4 videos with Viser.
* **Reference** - :doc:`../api/index` lists public and semi-public entry
  points.

.. _An Unconstrained Convex Formulation of Compliant Contact: https://arxiv.org/abs/2110.10107
.. _Warp: https://nvidia.github.io/warp/stable/
.. _Drake: https://github.com/RobotLocomotion/drake
