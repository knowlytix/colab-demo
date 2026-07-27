# KnowlytiX Colab demos

Runnable examples for the KnowlytiX Knowledge Abstraction Layer (KAL),
Geometric Memory System (GMS), knowledge tooling, and benchmark utilities.
The repository is packaged as one installable Python project so a Colab
runtime can install the complete demo environment directly from GitHub.

## Repository layout

```text
.
├── knowlytix/              # Core, KAL, knowledge, benchmark, and harness modules
├── knowlytix_demo/         # Demo helpers and bundled fixture data
├── notebooks/              # Runnable Google Colab/Jupyter notebooks
├── pyproject.toml          # Package metadata and runtime dependencies
└── README.md
```

## KAL + GMS end-to-end demo

[`notebooks/kal_gms_e2e_demo.ipynb`](notebooks/kal_gms_e2e_demo.ipynb)
shows how KAL combines triples from multiple sources, how GMS trains a
geometric knowledge graph, and how grounded multi-hop answers can be
distinguished from unsupported geometric guesses. Later sections compare
grounded answers with an LLM-only baseline and optionally demonstrate live
federation with Neo4j.

[![Open the KAL + GMS demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QjSnquvpG8wm8T5NfS_tDJmibV_RYYA2?usp=sharing)

The linked public Colab is the easiest way to run the demo. A copy is also
kept in this repository for versioning.

### 1. Use a compatible runtime

The included GMS extensions are compiled for 64-bit Linux and CPython 3.12.
In Colab, select **Runtime → Change runtime type → Python 3**. The notebook
checks the interpreter and platform during setup.

For a local run, use a Linux x86-64 environment with Python 3.12.

### 2. Install the repository and all dependencies

The notebook's setup cell runs:

```python
%pip install --upgrade --no-cache-dir \
  "git+https://github.com/knowlytix/colab-demo.git"
```

This installs the checked-in KnowlytiX packages, the compiled GMS extensions,
the demo fixtures, and every third-party dependency declared in
`pyproject.toml`. Restart the runtime if Colab asks you to do so after the
installation.

For a local editable checkout, use:

```bash
git clone https://github.com/knowlytix/colab-demo.git
cd colab-demo
python3.12 -m pip install -e .
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

## Security

Never commit a KnowlytiX license, Anthropic key, or Neo4j credentials. Use
Colab Secrets or environment variables. The repository contains public
license-verification keys only; it does not contain a user license.
