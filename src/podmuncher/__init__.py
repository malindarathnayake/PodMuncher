"""Podmuncher package and lightweight command entry point."""

__version__ = "0.1.0.dev0"
__all__ = ["__version__", "main"]


def _client_startup() -> None:
    """CLI client-startup hook: imports the composition only when a command actually runs."""
    from . import bootstrap

    bootstrap.client_startup()


def main(argv: list[str] | None = None) -> int:
    """Run the command line; the CLI module is imported only when invoked.

    The client-startup hook is registered here (SPEC_GAP-11).  The CLI calls it only for a
    non-help/version invocation, so ``--help``/``--version`` never import the composition.
    """
    from . import cli

    cli.set_client_startup(_client_startup)
    return cli.main(argv)
