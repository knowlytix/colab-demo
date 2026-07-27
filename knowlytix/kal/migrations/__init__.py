"""Alembic migrations for KAL-owned tables.

v0.2.0 ships two migrations:

- ``kal_001`` — external_id_registry table (KAL spec §6.4 Layer 4)
- ``kal_002`` — kal_connections registry table

Both create rows under the schema named by the ``KAL_DB_SCHEMA`` env
var (default ``kal``). Tenant FKs are deliberately omitted — the
package doesn't know what a "tenant" means in the consuming app;
referential integrity is the consumer's responsibility.

Two ways to run these migrations:

1. **Standalone (the package's own Alembic env)**: point Alembic at
   the directory this module lives in. Example::

       alembic -c knowlytix/kal/migrations/alembic.ini upgrade head

2. **Merged into a host app's migration tree**: include the package's
   migration directory as a second ``version_locations`` entry in
   the host's ``alembic.ini``. The migrations use a distinct
   ``branch_labels=("knowlytix_kal",)`` on the head so they don't
   conflict with the host's revision DAG.
"""
