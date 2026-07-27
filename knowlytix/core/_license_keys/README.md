<!-- SPDX-License-Identifier: LicenseRef-Knowlytix-GMS-EULA -->
# `_license_keys/` — trusted rotation / self-serve public keys

This directory ships inside the `knowlytix-core` wheel at
`knowlytix/core/_license_keys/`. It holds **additional** trusted RSA public
keys (JWKS-style) so the license signing keypair can rotate without breaking
already-issued licenses.

## How verification uses this directory

At runtime `knowlytix/core/_license.py::_load_public_keys()` builds the trusted
set `{kid: pem}` from two sources:

| Source | kid | Role |
|---|---|---|
| `knowlytix/core/knowlytix-public.pem` | `knowlytix-public` | **primary** (internal, signs `enterprise`) |
| `knowlytix/core/_license_keys/<id>.pem` | `<id>` | rotation / self-serve |

A license verifies if its RS256 signature matches **any one** trusted key. The
JWT `kid` header only orders which key is tried first — it never authorizes a
bad signature. The `enterprise` tier is bound to the primary kid, so a
rotation/self-serve key may sign `developer` but never a commercial key (see
`distribution/license_key_rotation_plan.md` §2).

**Reserved name:** a file named `knowlytix-public.pem` here is **ignored** —
the primary key ships only at the top-level path. The loader refuses to let the
rotation dir alias or override the primary.

**Currently bundled:** `selfserve.pem` (kid `selfserve`, GMS-260) — the public
half of the self-serve trial-issuer key. It signs `developer`-only licenses
(the GMS-261 Worker / `provision.py --kid selfserve`); the private half lives in
the secret store, never git.

## Adding a rotation key (issuer-side runbook)

1. Generate an RSA-2048 keypair. Keep the **private** key in the secret store
   (never commit it — `*.pem` is gitignored except declared public keys).
2. Drop the **public** key here as `<kid>.pem` (e.g. `2027-rotation.pem`). The
   kid is the filename without `.pem`.
3. Cut a `knowlytix-core` release — the new wheel now trusts both old and new
   keys. The packaging plumbing ships this dir automatically in both the pure
   (`hatch_build.py::_configure_pure`) and Nuitka-compiled (`build.sh` stages it
   into the wheel) layouts; `verify_wheels.sh` asserts it is present.
4. Start signing new/renewed licenses with the new private key via
   `provisioning/provision.py --kid <kid>`.

## Retiring an old key

A key is safe to drop only once **every** license signed with it has expired
(there is no runtime revocation — a JWT verifies until its `expires` date).
Run `provisioning/key_status.py` to see live counts and the latest expiry per
kid, then remove the retired `<kid>.pem` in a later release.
