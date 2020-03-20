import os
import sys

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:

    dataset = 'RHD'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'data')
    data_loc = '/media/yjs'
    output_dir = os.path.join(root_dir, 'output')
    util_dir = os.path.join(root_dir, 'utils')
    vis_dir = os.path.join(output_dir, 'vis')
    result_dir = os.path.join(output_dir, 'result')

    vis_interval = 500

    fitting_iter_max = 2500
    ncomps = 45

cfg = Config()
sys.path.insert(0, cfg.root_dir)
add_pypath(cfg.data_dir)
add_pypath(cfg.util_dir)


