Viser Viewer
============

The Viser viewer is an interactive browser-based visualization entry point for
SAP Warp scenes. It loads the same YAML scene files used by ``benchmark.py``,
steps :class:`~sim.solver_sap.SolverSAP`, and streams visual geometry through
the official ``viser`` package.

Quick Run
---------

Launch the default scene:

.. code-block:: shell

   uv run python -m viewer.viser.sap_viewer --num-worlds 1

Launch a specific scene:

.. code-block:: shell

   uv run python -m viewer.viser.sap_viewer --scene assets/yaml/multi_joints.yaml

The command prints a local URL such as ``http://localhost:8080``. Open that URL
in a browser to inspect the simulation.

Useful Runtime Flags
--------------------

``--scene``
   YAML scene file. Defaults to ``assets/yaml/unitree_g1_usd.yaml``.

``--duration`` and ``--frames``
   Stop after a simulated duration or exact frame count. Leave both unset for
   an interactive run.

``--dt``
   Timestep override. Defaults to ``simulation.dt`` in the scene file.

``--num-worlds``
   Number of replicated worlds. Defaults to ``simulation.num_worlds``.

``--viewer-fps``
   Target viewer display update rate. Higher values can look smoother but
   increase viewer overhead. The actual achieved viewer FPS depends on scene
   complexity, browser rendering cost, and recording overhead.

``--substeps-per-frame``
   Number of simulation steps between viewer updates. Defaults to ``1``.
   Increasing this can improve simulation throughput at the cost of less
   frequent visual updates.

``--disable-cuda-graph``
   Disable CUDA graph replay and step through the direct Python path.

MP4 Recording
-------------

Record the Viser browser output to an MP4 file:

.. code-block:: shell

   uv run python -m viewer.viser.sap_viewer \
     --scene assets/yaml/multi_joints.yaml \
     --duration 5 \
     --record-mp4 outputs/multi_joints.mp4

Recording options:

``--record-mp4``
   Output MP4 path.

``--record-fps``
   Recording FPS. Defaults to ``--viewer-fps``.

``--record-width`` and ``--record-height``
   Browser capture size in pixels. Defaults to ``1280x720``.

``--record-webgl``
   WebGL backend for the recording browser. The default ``gpu`` mode requests
   hardware WebGL. Use ``swiftshader`` only as a software fallback if headless
   GPU WebGL is unavailable on the machine.

Notes
-----

The viewer uses visual meshes when the scene loader provides them. Primitive
inline shapes can set a display color with a top-level ``color`` field:

.. code-block:: yaml

   shapes:
     - type: box
       hx: 0.1
       hy: 0.1
       hz: 0.75
       color: [0.1, 0.75, 1.0]

The viewer is not the benchmark path. For timing-only measurements, prefer
``benchmark.py``.
