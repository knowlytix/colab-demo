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
| KAL + GMS end-to-end | [`notebooks/kal_gms_e2e_demo.ipynb`](notebooks/kal_gms_e2e_demo.ipynb) | `kal-gms` |

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

[`notebooks/kal_gms_e2e_demo.ipynb`](notebooks/kal_gms_e2e_demo.ipynb)
shows how KAL combines triples from multiple sources, how GMS trains a
geometric knowledge graph, and how grounded multi-hop answers can be
distinguished from unsupported geometric guesses. Later sections compare
grounded answers with an LLM-only baseline and optionally demonstrate live
federation with Neo4j.

[![Open the KAL + GMS demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knowlytix/colab-demo/blob/main/notebooks/kal_gms_e2e_demo.ipynb)

The Colab link loads the notebook directly from this repository's `main`
branch, so it always reflects the latest checked-in version.

### 1. Use a compatible runtime

The included GMS extensions are compiled for 64-bit Linux and CPython 3.12.
In Colab, select **Runtime → Change runtime type → Python 3**. The notebook
checks the interpreter and platform during setup.

For a local run, use a Linux x86-64 environment with Python 3.12.

### 2. Store this demo's dependencies in Google Drive

The notebook does not assume that it is saved in a particular Drive directory.
Its setup cell mounts the current user's Google Drive, creates a dedicated
folder relative to **My Drive**, and installs the `kal-gms` environment there:

```text
My Drive/
└── KnowlytiX/
    └── colab-demo/
        └── python-3.12/
            └── site-packages/
```

The folder is controlled near the top of the setup cell:

```python
DRIVE_FOLDER = Path("KnowlytiX") / "colab-demo"
REFRESH_DRIVE_DEPENDENCIES = False
```

Change `DRIVE_FOLDER` to another safe path relative to My Drive if desired.
It is a dependency-storage location, not the notebook's location. The first
run downloads the checked-in KnowlytiX packages, compiled GMS extensions, demo
fixtures, and all third-party packages assigned to the `kal-gms` extra.
Subsequent sessions reuse that persisted environment and add it to
`sys.path`.

Set `REFRESH_DRIVE_DEPENDENCIES = True` for one run whenever you want to pull
repository or dependency updates, then return it to `False`. If Colab asks for
Drive access, authorize the mount for the Google account in which you want the
environment stored.

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

The `KAL + GMS README demo` workflow checks the same path described above on
Ubuntu x86-64 with Python 3.12.

- On relevant pushes and pull requests, it installs `.[kal-gms]`, builds the
  package, and validates every notebook code cell without making paid API calls.
- A manual **Run workflow** execution runs the entire notebook top-to-bottom,
  including the Anthropic and Neo4j sections. Neo4j runs in a disposable
  service container.

Before running the full workflow, add `KNOWLYTIX_LICENSE_KEY` and
`ANTHROPIC_API_KEY` under **Repository settings → Secrets and variables →
Actions**. You may also add `GMS_LLM_MODEL`; otherwise the notebook uses its
documented default. The workflow never prints secret values.

## Security

Never commit a KnowlytiX license, Anthropic key, or Neo4j credentials. Use
Colab Secrets or environment variables. The repository contains public
license-verification keys only; it does not contain a user license.
