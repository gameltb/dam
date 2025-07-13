from domarkx.cli import cli_app, load_actions

def setup_test_app():
    load_actions()
    return cli_app
