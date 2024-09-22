import os,sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import project_statics

MASSIVE_DATA_BASE = (
  "/content/1.0/data/en-US.jsonl"
)

from utils import parse_ourData_newformat

# raw file path, save destination path
parse_ourData_newformat(MASSIVE_DATA_BASE, project_statics.SFID_pickle_files)