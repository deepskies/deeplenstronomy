from cleo import Application
import sys

from client.config_commands import FullRunCommand

deeplens_app = Application()
deeplens_app.add(FullRunCommand())

def main() -> int:
    return deeplens_app.run()

if __name__ == '__main__':
    sys.exit(main())
