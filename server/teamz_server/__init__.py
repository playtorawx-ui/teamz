"""teamz FastAPI orchestrator.

Public surface:

    from teamz_server import create_app
    app = create_app()

Everything else (registry, router, ledger, keys, memory) is injectable so the
whole stack is testable without a network.
"""

from .app import create_app

__all__ = ["create_app"]
__version__ = "0.2.0"
