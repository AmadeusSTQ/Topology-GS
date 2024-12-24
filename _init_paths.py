from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
add_path(this_dir)

# lib_path0 = osp.join(this_dir, 'experiments')
# add_path(lib_path0)
#
# lib_path1 = osp.join(this_dir, 'gaussian_renderer')
# add_path(lib_path1)
#
# lib_path2 = osp.join(this_dir, 'scene')
# add_path(lib_path2)
#
# lib_path3 = osp.join(this_dir, 'TOGE_users')
# add_path(lib_path3)
#
# lib_path4 = osp.join(this_dir, 'utils')
# add_path(lib_path4)
