.. _api-benchmark:
.. _api-load-sap-scene-config:
.. _api-load-sap-scene:
.. _api-sap-loaded-scene:
.. _api-sap-unsupported-scene-feature:
.. _api-sap-model:
.. _api-sap-state:
.. _api-sap-state-joint-q:
.. _api-sap-state-joint-qd:
.. _api-sap-state-body-f:
.. _api-sap-control:
.. _api-sap-control-joint-f:
.. _api-sap-model-from-newton:
.. _api-sap-state-from-newton:
.. _api-sap-control-from-newton:
.. _api-sap-collision-model:
.. _api-sap-collision-state-from-state:
.. _api-sap-collision-pipeline:
.. _api-sap-estimate-rigid-contact-max:
.. _api-sap-contacts:
.. _api-solver-sap:
.. _api-solver-sap-step:
.. _api-solver-sap-step-state-out:
.. _api-solver-sap-step-dt:
.. _api-solver-sap-contact-weight-mode:
.. _api-solver-sap-line-search-variant:
.. _api-solver-sap-line-search-monotone-decay:
.. _api-solver-sap-line-search-armijo-decay:
.. _api-solver-sap-line-search-exact-root:
.. _api-solver-sap-armijo-c:
.. _api-solver-sap-rho:
.. _api-solver-sap-line-search-max-iterations:
.. _api-solver-sap-graph-conditional:
.. _api-solver-sap-position-integration:
.. _api-solver-sap-position-integration-midpoint:
.. _api-solver-sap-position-integration-sap-euler:
.. _api-contact-jacobian:
.. _api-contact-jacobian-compute:
.. _api-contact-jacobian-result:
.. _api-contact-jacobian-result-contact-env-jacobian:
.. _api-contact-jacobian-result-contact-env-phi0:
.. _api-contact-jacobian-result-contact-env-mu:
.. _api-contact-jacobian-result-contact-env-stiffness:
.. _api-contact-jacobian-result-contact-env-tau-d:
.. _api-contact-jacobian-result-dynamics-matrix-env:
.. _api-contact-solve:
.. _api-contact-solve-solve:
.. _api-free-motion:
.. _api-free-motion-compute:
.. _api-free-motion-assemble-dynamics-matrix:
.. _api-free-motion-result:
.. _api-free-motion-result-v-star:
.. _api-free-motion-result-vdot:
.. _api-free-motion-result-dynamics-matrix:
.. _api-free-motion-joint-q-input:
.. _api-free-motion-joint-qd-sap-input:
.. _api-free-motion-joint-f-sap-input:
.. _api-free-motion-body-f-ext-s:
.. _api-free-motion-free-motion-joint-qd-sap:
.. _api-sap-helpers:
.. _api-contact-jacobian-module:

API Reference
=============

This page is a map of the SAP Warp surface you normally touch from Python. It
does not try to list every Newton-derived helper or Warp kernel; those modules
remain implementation details unless they are needed to understand the SAP
runtime path.

Read the reference in the same order as a simulation step:

1. Load a scene with :func:`sim.loader.scene.load_sap_scene`.
2. Keep the returned :class:`sim.sap_runtime.SapModel`,
   :class:`sim.sap_runtime.SapState`, and :class:`sim.sap_runtime.SapControl`
   objects as the solver boundary data.
3. Build :class:`sim.collision.pipeline.SapCollisionPipeline` from the collision
   model and call :meth:`sim.collision.pipeline.SapCollisionPipeline.collide`
   whenever body poses change.
4. Pass the active contacts into :meth:`sim.solver_sap.SolverSAP.step`.

The generated pages below give each documented object its own entry, while the
short notes in this page explain why the object exists in the end-to-end loop.

Benchmark Entry Points
----------------------

Use these when you want to run the repository benchmark exactly as the command
line tool does. They are thin orchestration helpers around the same loader,
collision, and solver APIs shown later on this page.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   benchmark.build_parser
   benchmark.run_native

Scene Loader
------------

The loader is the bridge from tutorial examples and YAML/JSON scene files into
runtime arrays. Start here when you need a model, initial state, control, and
collision model without constructing Warp arrays by hand.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   sim.loader.scene.load_sap_scene_config
   sim.loader.scene.load_sap_scene
   sim.loader.scene.SapLoadedScene
   sim.loader.scene.SapUnsupportedSceneFeature

Runtime Data
------------

Runtime data objects are plain containers around Warp arrays. ``SapModel`` is
the immutable topology and default data, while ``SapState`` and ``SapControl``
are the mutable boundary objects you pass to a timestep.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   sim.sap_runtime.SapModel
   sim.sap_runtime.SapState
   sim.sap_runtime.SapControl
   sim.sap_runtime.SapContacts
   sim.sap_runtime.sap_model_from_newton
   sim.sap_runtime.sap_state_from_newton
   sim.sap_runtime.sap_control_from_newton
   sim.sap_runtime.sap_contacts_from_newton

Runtime Methods
---------------

These methods allocate or reset runtime buffers. They are intentionally small,
but they define the normal ownership pattern: create states and controls from
the model, then clear only the buffers that should not carry over between
steps.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   sim.sap_runtime.SapModel.state
   sim.sap_runtime.SapModel.control
   sim.sap_runtime.SapState.clear_forces
   sim.sap_runtime.SapControl.clear

Collision
---------

Collision APIs own the shape-side model and the active contact buffer. The
pipeline is separate from the solver so callers can choose exactly when contact
generation runs. The implementation deliberately tracks Newton's collision data
model for compatibility with imported assets and geometry behavior, then adapts
the generated contacts into SAP-owned buffers. Hydroelastic contact support is
in development; the documented API here covers the rigid-contact path.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   sim.resources.collision_model.SapCollisionModel
   sim.resources.collision_model.SapCollisionState
   sim.resources.collision_model.sap_collision_state_from_state
   sim.collision.pipeline.SapCollisionPipeline
   sim.collision.pipeline.sap_estimate_rigid_contact_max
   sim.collision.pipeline.sap_normalize_broad_phase_mode
   sim.collision.contacts.SapContacts

Collision Methods
-----------------

Use ``contacts`` once to allocate the output buffer, then call ``collide`` each
time current body poses should produce a new active contact set.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   sim.collision.pipeline.SapCollisionPipeline.contacts
   sim.collision.pipeline.SapCollisionPipeline.collide

Solver
------

The solver section contains the high-level timestepper and the three internal
SAP stages that are useful when debugging or validating the solve: free motion,
contact Jacobian assembly, and contact solve.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   sim.solver_sap.SolverSAP
   sim.contact_jacobian.SapContactJacobian
   sim.contact_jacobian.SapContactJacobianResult
   sim.contact_solve.SapContactSolve
   sim.contact_solve.SapContactSolveResult
   sim.free_motion.SapFreeMotion
   sim.free_motion.SapFreeMotionResult

Solver Methods
--------------

For normal use, :meth:`sim.solver_sap.SolverSAP.step` is the method you call.
The stage-specific methods are listed so advanced users can inspect the same
data flow one stage at a time.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   sim.solver_sap.SolverSAP.step
   sim.free_motion.SapFreeMotion.compute
   sim.contact_jacobian.SapContactJacobian.compute
   sim.contact_solve.SapContactSolve.solve

Modules
-------

These modules are documented as modules because they collect helper kernels or
stage-specific data that is easier to understand in context than as dozens of
low-level entries.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   sim.sap_helpers
   sim.contact_jacobian

Minimal Program
---------------

The shortest complete program follows the same sequence as the benchmark:

.. code-block:: python

   import warp as wp

   from sim.collision.pipeline import SapCollisionPipeline
   from sim.loader.scene import load_sap_scene
   from sim.resources.collision_model import sap_collision_state_from_state
   from sim.solver_sap import SolverSAP

   num_worlds = 1
   max_rigid_contact_per_env = 48
   rigid_contact_capacity = max_rigid_contact_per_env * num_worlds
   device = wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")

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
   )
   collision = SapCollisionPipeline(
       scene.collision_model,
       rigid_contact_max=rigid_contact_capacity,
   )
   contacts = collision.contacts()

   state_0 = scene.sap_state
   state_1 = scene.sap_model.state()
   control = scene.sap_control

   state_0.clear_forces()
   collision.collide(sap_collision_state_from_state(state_0), contacts)
   solver.step(state_0, state_1, control, contacts, 0.003)
