import os
import sys
from pathlib import Path


def add_project_root_to_sys_path():
    sys.path.append(str(Path(os.getcwd()).parent.parent.absolute()))
