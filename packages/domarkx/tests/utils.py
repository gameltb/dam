from domarkx.cli import cli_app, load_actions
from domarkx.config import settings


def setup_test_app():
    load_actions(settings)
    return cli_app
