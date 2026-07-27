"""Alembic environment for standalone KAL migrations.

Used when the package is driven via its own ``alembic.ini``
(option 1 in ``knowlytix.kal.migrations.__init__``). Consuming apps
that merge KAL migrations into their own tree use their own env.py
and don't import this module.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

# Importing model modules so ``target_metadata`` knows about every
# table KAL owns. Order doesn't matter — both modules attach to the
# same ``KalBase.metadata``.
from knowlytix.kal._db import KalBase
from knowlytix.kal.connections import models as _conn_models  # noqa: F401
from knowlytix.kal.external_id_registry import models as _ext_models  # noqa: F401

config = context.config

# Allow the database URL to come from the env so the alembic.ini
# stays free of secrets.
db_url = os.environ.get("KAL_DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

target_metadata = KalBase.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connect and apply."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
