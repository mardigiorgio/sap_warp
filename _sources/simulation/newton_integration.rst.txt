Newton Integration
==================

``sap_with_newton.py`` demonstrates the direct Newton integration path. The
example builds a Newton cartpole model, displays it with Newton's viewer, and
uses :class:`~sim.solver_sap.SolverSAP` as the timestepper.

The integration is intentionally thin: Newton still owns the model, state,
control, viewer, and optional collision buffers. SAP Warp wraps those arrays at
the solver boundary and writes the next Newton state in place.

Run the Example
---------------

Newton is an optional, on-demand dependency for this repository. Run the demo
with ``uv run --with`` so the package is available for this command without
adding it to ``pyproject.toml``:

.. code-block:: shell

   uv run --frozen --with 'newton[examples]' python sap_with_newton.py

The defaults are the interactive GL viewer, SAP as the solver backend, and one
Newton world. The script builds the Newton cartpole model directly in Python
rather than loading one of the repository YAML scene files.

Run a short headless smoke test:

.. code-block:: shell

   uv run --frozen --with 'newton[examples]' python sap_with_newton.py \
     --viewer null \
     --num-frames 10

Run the same scene through Newton's solver for comparison:

.. code-block:: shell

   uv run --frozen --with 'newton[examples]' python sap_with_newton.py \
     --viewer null \
     --num-frames 10 \
     --solver newton

Data Boundary
-------------

The adapter functions in :mod:`sim.sap_runtime` turn Newton runtime objects into
SAP-facing containers:

* :func:`~sim.sap_runtime.sap_model_from_newton` wraps a Newton model as a
  :class:`~sim.sap_runtime.SapModel`.
* :func:`~sim.sap_runtime.sap_state_from_newton` wraps a Newton state as a
  :class:`~sim.sap_runtime.SapState`.
* :func:`~sim.sap_runtime.sap_control_from_newton` wraps a Newton control object
  as a :class:`~sim.sap_runtime.SapControl`.
* Newton contact buffers can be passed directly to
  :meth:`~sim.solver_sap.SolverSAP.step`; the solver converts them with
  :func:`~sim.sap_runtime.sap_contacts_from_newton` when needed.

These wrappers do not replace Newton as the owner of the simulation data. They
record how SAP should interpret the same Warp arrays, including public Newton
generalized velocity and force ordering. Before the solve, ``SolverSAP`` maps
public Newton ordering into SAP's internal ordering; after integration, it maps
the output velocity back to the public state layout. The SAP runtime wrapper
data structures are based on
`Newton <https://github.com/newton-physics/newton>`__'s model, state, control,
and contact container code so this adapter can stay thin and compatible with
Newton-owned arrays.

Minimal Pattern
---------------

The core setup in ``sap_with_newton.py`` is:

.. code-block:: python

   model = create_newton_cartpole_model(args, device)
   state_0 = model.state()
   state_1 = model.state()
   control = model.control()
   contacts = model.contacts() if args.collision == "newton" else None

   newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
   state_1.assign(state_0)

   sap_model = sap_model_from_newton(model)
   solver = SolverSAP(
       sap_model,
       max_rigid_contact=args.contact_cap,
       max_iterations=args.solver_iterations,
       contact_tau_d=args.contact_tau_d,
       contact_preset_variant=args.contact_preset,
       line_search_variant=args.line_search,
   )
   sap_state_0 = sap_state_from_newton(state_0)
   sap_state_1 = sap_state_from_newton(state_1)
   sap_control = sap_control_from_newton(control)

``newton.eval_fk`` initializes the body transforms from the Newton generalized
coordinates before SAP takes the first step. The two Newton states remain the
double buffers used by the viewer and by either solver backend.

Step Loop
---------

Each rendered frame can contain multiple simulation substeps. The SAP path keeps
Newton's normal frame structure:

.. code-block:: python

   state_0.clear_forces()
   viewer.apply_forces(state_0)

   if contacts is not None:
       model.collide(state_0, contacts)

   solver.step(sap_state_0, sap_state_1, sap_control, contacts, sim_dt)
   sap_state_0, sap_state_1 = sap_state_1, sap_state_0
   state_0, state_1 = state_1, state_0

The SAP state wrappers point at the same arrays as ``state_0`` and ``state_1``.
After every step, swap both wrapper buffers and Newton buffers together. Newton's
viewer can then render ``state_0`` without any copy back from SAP.

Contacts
--------

For the cartpole example, contacts are disabled by default because the model is
useful as a pure articulated dynamics test:

.. code-block:: shell

   uv run --frozen --with 'newton[examples]' python sap_with_newton.py --collision none

For contact experiments, ask Newton to allocate and refresh contacts:

.. code-block:: shell

   uv run --frozen --with 'newton[examples]' python sap_with_newton.py --collision newton --contact-cap 128

``--contact-cap`` is the per-world rigid-contact capacity passed to
``SolverSAP``. When using replicated Newton worlds, keep this value large enough
for the maximum active contacts in one world; the solver sizes the total contact
work from that per-world cap and the Newton model's world count.

Solver Controls
---------------

The example exposes the main SAP knobs without changing Newton's outer loop:

* ``--solver-iterations`` sets the maximum SAP contact solve iterations.
* ``--contact-preset`` selects the precision and contact Jacobian preset.
* ``--line-search`` selects the SAP line-search variant.
* ``--contact-tau-d`` provides the fallback contact dissipation time scale.
* ``--world-count`` replicates the Newton cartpole model before conversion.

Use this pattern when you want Newton's asset loading, replication, viewer, or
interaction utilities, but want the SAP Warp solver to advance the articulated
state.
