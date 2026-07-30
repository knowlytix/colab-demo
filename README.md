# KnowlytiX Colab demos

Runnable examples for the KnowlytiX Knowledge Abstraction Layer (KAL),
Geometric Memory System (GMS), knowledge tooling, and benchmark utilities.
The repository can contain multiple demos. Each notebook declares its own
dependency extra, so you install only the dependencies needed for the notebook
you want to run.

## Repository layout

```text
.
├── knowlytix/              # Core, KAL, knowledge, benchmark, and harness modules
├── knowlytix_demo/         # Demo helpers and bundled fixture data
├── notebooks/              # Runnable Google Colab/Jupyter notebooks
├── pyproject.toml          # Package metadata and runtime dependencies
└── README.md
```

## Choose a demo

Do not install every possible demo dependency by default. Select a notebook
below and use its matching install command. As more demos are added, each one
will receive a separate dependency extra and a row in this table.

| Demo | Notebook | Dependency extra |
|---|---|---|
| KAL + GMS end-to-end | [KAL + GMS — end-to-end demo](notebooks/kal_gms_e2e_demo.ipynb) | `kal-gms` |

For example, installing the repository without an extra installs only the
shared KnowlytiX source:

```bash
python -m pip install \
  "git+https://github.com/knowlytix/colab-demo.git"
```

Installing with `[kal-gms]` adds the third-party dependencies required by the
KAL + GMS notebook:

```bash
python -m pip install \
  "knowlytix-colab-demo[kal-gms] @ git+https://github.com/knowlytix/colab-demo.git"
```

## KAL + GMS end-to-end demo

[KAL + GMS — end-to-end demo](notebooks/kal_gms_e2e_demo.ipynb)
shows how KAL combines triples from multiple sources, how GMS trains a
geometric knowledge graph, and how grounded multi-hop answers can be
distinguished from unsupported geometric guesses. Later sections compare
grounded answers with an LLM-only baseline and optionally demonstrate live
federation with Neo4j.

[![Open KAL + GMS — end-to-end demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knowlytix/colab-demo/blob/main/notebooks/kal_gms_e2e_demo.ipynb)

The Colab link loads the notebook directly from this repository's `main`
branch, so it always reflects the latest checked-in version.

### 1. Use a compatible runtime

The included GMS extensions are compiled for 64-bit Linux and CPython 3.12.
In Colab, select **Runtime → Change runtime type → Python 3**. The notebook
checks the interpreter and platform during setup.

For a local run, use a Linux x86-64 environment with Python 3.12.

### 2. Install dependencies with a persistent Drive cache

The notebook does not assume that it is saved in a particular Drive directory.
By default, its setup cell installs packages onto Colab's fast local runtime
disk and uses a dedicated Google Drive folder only as pip's persistent download
cache:

```text
My Drive/
└── KnowlytiX/
    └── colab-demo/
        └── pip-cache/
```

The behavior is controlled near the top of the setup cell:

```python
DRIVE_FOLDER = Path("KnowlytiX") / "colab-demo"
USE_DRIVE_PIP_CACHE = True
REFRESH_PACKAGE = False
```

Change `DRIVE_FOLDER` to another safe path relative to My Drive if desired.
It is a download-cache location, not the notebook's location. Python imports
packages from the active Colab runtime; it does not import thousands of package
files through the slower mounted-Drive filesystem. The first session downloads
missing packages. Later sessions can reuse cached archives, while packages that
Colab already provides are left in place. Rerunning the cell in the same
runtime is skipped automatically.

Set `REFRESH_PACKAGE = True` for one run when you want to pull repository or
dependency updates, then return it to `False`. Set `USE_DRIVE_PIP_CACHE =
False` to avoid mounting Drive and use Colab's temporary local pip cache
instead. If Colab asks for Drive access, authorize the mount for the Google
account in which you want the cache stored.

#### Dependency persistence options

| Option | How to use it | Tradeoff |
|---|---|---|
| **Local install + Drive pip cache (default)** | Keep `USE_DRIVE_PIP_CACHE = True`. | Best balance: fast imports and faster repeat sessions. Packages are reinstalled into each new runtime, but downloads are reused. |
| **Local install only** | Set `USE_DRIVE_PIP_CACHE = False`. | Fastest setup when Colab already has most packages and no Drive permission is needed, but missing packages must be downloaded after every runtime reset. |
| **Drive wheelhouse** | Pre-download pinned wheels to Drive, then install with `pip --no-index --find-links=<wheelhouse>`. | More reproducible and can work without PyPI after preparation, but requires maintaining a Python/platform-specific wheel set and still installs locally each session. |
| **Full environment in Drive** | Install with `pip --target=<Drive folder>` and add it to `sys.path`. | Persists installed files, but is not recommended: creating and importing thousands of small files through mounted Drive can take many minutes and can be less reliable. This was the notebook's previous behavior. |
| **Prebuilt custom runtime/container** | Run the notebook against a local Jupyter runtime or managed image with dependencies preinstalled. | Fastest startup and strongest reproducibility, but it is not a standard hosted-Colab workflow and requires separate infrastructure. |

Outside Colab, the same setup cell installs into the active Python environment
instead. For a local editable checkout, use:

```bash
git clone https://github.com/knowlytix/colab-demo.git
cd colab-demo
python3.12 -m pip install -e ".[kal-gms]"
```

### 3. Add a KnowlytiX license

Obtain a KnowlytiX license from [knowlytix.ai](https://knowlytix.ai/). In
Colab, open the **Secrets** panel (the key icon in the left sidebar), create a
secret named `KNOWLYTIX_LICENSE_KEY`, paste the license value, and enable
notebook access. Do not place the license in a code cell or commit it to Git.

The first notebook cell reads that secret, exports it only in the current
runtime, and writes it to the standard runtime path:

```text
~/.knowlytix/license.key
```

Run this cell before importing any `knowlytix` module.

### 4. Run the license-only sections

Run the notebook from the top. Sections 1–5 use the bundled JSONL and Markdown
fixtures and require only the installed repository plus your KnowlytiX
license. They do not require an external database or LLM key.

### 5. Enable the Anthropic sections

Sections 6–7 call an LLM to parse, phrase, and compare answers. To run them,
provide your own Anthropic API key:

1. In Colab Secrets, add `ANTHROPIC_API_KEY` and enable notebook access.
2. Optionally add `GMS_LLM_MODEL`; if omitted, the notebook uses the
   Anthropic model configured by the bundled preflight helper.
3. Run the short configuration cell immediately before section 6.

API usage is billed by Anthropic under your account. Skip sections 6–7 if you
do not want to make external LLM calls.

### 6. Enable the Neo4j federation section

Section 8 is optional and needs a Neo4j database reachable from the Colab
runtime. A database running only on your laptop at `localhost` is not
reachable from hosted Colab. Use your own hosted Neo4j instance (for example,
Neo4j Aura) and add these Colab Secrets:

```text
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
```

Grant notebook access to all three secrets, then run the Neo4j configuration
cell in section 8. The demo uses `MERGE`, so rerunning its seed cells is
idempotent, but it will add the demo nodes and relationships to the selected
database. Use a disposable database if you do not want demo data mixed with
existing data.

### 7. Recommended execution order

1. Open the public Colab link.
2. Confirm the runtime uses Python 3.12 on Linux x86-64.
3. Add `KNOWLYTIX_LICENSE_KEY` to Colab Secrets.
4. Run setup and sections 1–5 in order.
5. Add `ANTHROPIC_API_KEY` and run sections 6–7 if desired.
6. Add the three Neo4j secrets and run section 8 if desired.

Generated `_store_nb_*` directories are runtime artifacts. They are ignored by
Git and can be deleted between runs.

## GitHub Actions verification

The public-safe `KAL + GMS README demo` workflow checks the setup path on
Ubuntu x86-64 with Python 3.12. On relevant pushes, pull requests, and manual
dispatches, it:

1. Installs `.[kal-gms]`.
2. Builds the installable package.
3. Validates every notebook code cell.

The workflow does not receive a KnowlytiX license, API key, or database
credential. It does not execute paid Anthropic calls, start Neo4j, run licensed
notebook sections, or upload an executed notebook. Full demo execution remains
an interactive user action in Colab with credentials supplied by that user.

## Security

Never commit a KnowlytiX license, Anthropic key, or Neo4j credentials. Use
Colab Secrets or environment variables. The repository contains public
license-verification keys only; it does not contain a user license.
