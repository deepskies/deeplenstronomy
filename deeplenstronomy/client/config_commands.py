from cleo import Command, option

class FullRunCommand(Command):
    """
    Configures the deeplenstronomy application.

    config
        {--force : Force overwriting existing configuration file}
    """
    name = "run"
    description = "Run the deeplenstronomy application."
    arguments = []
    options = []

    def handle(self):
        # option = self.option('force')
        pass

    # 1. Import or create config file.
    # 2. Parse components of the config object.
    

