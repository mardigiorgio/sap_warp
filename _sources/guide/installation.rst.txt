Installation
============

SAP Warp is developed as a source checkout. Install from the repository root
with ``uv`` so the pinned lockfile, NVIDIA package index, and dependency groups
are used consistently.

Requirements
------------

.. list-table::
   :header-rows: 1

   * - Requirement
     - Version
     - Notes
   * - Python
     - 
     - Declared by ``pyproject.toml``.
   * - Package manager
     - ``uv``
     - Recommended for installing project dependencies.
   * - Warp
     - ``warp-lang>=1.7.0.dev20250823``
     - Resolved from the configured NVIDIA package index.
   * - CUDA GPU
     - Optional
     - Use ``--device cuda:0`` when available.
   * - Git
     - Required for shipped robot scenes
     - The loader uses sparse git checkouts for external USD/URDF assets.

Install Runtime Dependencies
----------------------------

.. code-block:: shell

   uv sync

This installs the runtime dependencies declared in ``pyproject.toml``. Install
the optional ``mesh`` extra when working with URDF mesh assets that require
``trimesh``:

.. code-block:: shell

   uv sync --extra mesh

Documentation dependencies live in the ``docs`` dependency group and are
installed on demand by ``uv run --group docs ...`` commands.

First Smoke Run
---------------

Run a short G1 scene:

.. code-block:: shell

   uv run python benchmark.py \
     --scene assets/yaml/unitree_g1_usd.yaml \
     --frames 2 \
     --num-worlds 1

Use ``--device cpu`` to force CPU execution or ``--device cuda:0`` to force the
first CUDA device. Omitting ``--device`` lets Warp choose the default device.

The G1 scene defaults to many replicated worlds for throughput measurement, so
``--num-worlds 1`` keeps the smoke run small. A successful run ends with a
summary line similar to:

.. code-block:: text

   scene=.../assets/yaml/unitree_g1_usd.yaml: device=... dt=0.003000 frames=2 num_worlds=1 ...

Asset Cache
-----------

The shipped Unitree scenes reference USD assets from
``https://github.com/newton-physics/newton-assets.git``. The scene loader stores
git-backed assets under ``SAP_WARP_ASSET_CACHE`` when that environment variable
is set, otherwise under ``~/.cache/sap_warp/assets``.

To keep project-local assets outside the home directory:

.. code-block:: shell

   SAP_WARP_ASSET_CACHE=.cache/assets \
     uv run python benchmark.py --scene assets/yaml/unitree_g1_usd.yaml --frames 1 --num-worlds 1

Use offline mode only after the referenced assets are already cached:

.. code-block:: shell

   SAP_WARP_ASSET_OFFLINE=1 uv run python benchmark.py --duration 0.005

Build the Docs
--------------

Build the Sphinx documentation locally:

.. code-block:: shell

   uv run --group docs sphinx-build -W -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` after the build completes.
Warnings are treated as errors, so broken references and stale examples should
be fixed before publishing documentation changes.

Troubleshooting
---------------

``source.git requires the git executable``
   Install ``git`` or use only local asset paths.

``Git asset ... is not cached and SAP_WARP_ASSET_OFFLINE is set``
   Disable ``SAP_WARP_ASSET_OFFLINE`` once so the loader can populate the
   cache, or point ``SAP_WARP_ASSET_CACHE`` at a cache that already contains
   the referenced revision.

``SolverSAP requires a scene with at least one joint DOF``
   Add a ``free``, ``revolute``, or ``prismatic`` joint to an inline scene, or
   load an asset that creates articulated DOFs.
