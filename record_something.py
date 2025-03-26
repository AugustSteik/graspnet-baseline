import inspect, re
import os
import json
import torch
import numpy as np
import jsbeautifier
from typing import Dict, List


opts = jsbeautifier.default_options()
opts.indent_size = 4

KEEP_KEYS = ('score', 'view', 'label')

def log_variable(log_name, in_name, in_var, out_path=None):
    """ Save the value of a variable and save it to a file.
    """
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), 'variable_logs', f'{log_name}.json')
    
    # Read existing data if the file exists
    if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                try:
                    out_data = json.load(f)
                except Exception:
                    os.remove(out_path)
                    out_data = {}
    else:
        out_data = {}
    
    if isinstance(in_var, List):
        if isinstance(in_var[0], torch.Tensor):
            out_list = [item.tolist() for item in in_var]
            in_var = {
                'dtype': str(in_var[0].dtype),
                'shape': [len(out_list)]+list(in_var[0].shape),
                'data': out_list,  # Convert tensor data to list
            }
        elif isinstance(in_var[0], np.ndarray):
            out_list = [item.tolist() for item in in_var]
            in_var = {
                'dtype': str(in_var[0].dtype),
                'length': [len(out_list)] + [len(in_var[0])],
                'data': out_list,  # Convert tensor data to list
            }
            
    elif isinstance(in_var, Dict):
        out_dict = {}
        for k, v in in_var.items():
            if not any(key in k.lower() for key in KEEP_KEYS):
                continue
            if isinstance(v, torch.Tensor):
                val = {k: {
                    'dtype': str(v.dtype),
                    'shape': list(v.shape),
                    'data': v.tolist(),  # Convert tensor data to list
                }}
                # out_data[in_name] = in_var
            elif isinstance(v, np.ndarray):
                val = {k: {
                    'dtype': str(v.dtype),
                    'length': len(v),
                    'data': v.tolist(),  # Convert np array data to list
                }}
            else:
                val = {k: v}
            out_dict.update(val)
        in_var = out_dict
        
    # Update the value for the given key
    elif isinstance(in_var, torch.Tensor):
        in_var = {
            'dtype': str(in_var.dtype),
            'shape': list(in_var.shape),
            'data': in_var.tolist(),  # Convert tensor data to list
        }
        # out_data[in_name] = in_var
    elif isinstance(in_var, np.ndarray):
        in_var = {
            'dtype': str(in_var.dtype),
            'length': len(in_var),
            'data': in_var.tolist(),  # Convert np array data to list
        }
    
        
    # else:
    out_data[in_name] = in_var
    
    # Write the updated data back to the file
    with open(out_path, 'w') as f:
        f.write(jsbeautifier.beautify(json.dumps(out_data), opts))
    return

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)
        
if __name__ == '__main__':
    again = torch.rand((12, 23 ,6))
    this = np.array([1, 2,3, 4])
    nert_one = 2389457280345.234234
    nert_one = {'this': torch.Tensor([1, 2, 3],),
                'that': np.array([1, 2, 3]),
                }
    nert_one = [again, again, again]
    namee = 'liasfjldf'
    # print(varname(this))
    varname(nert_one)
    # log_variable(namee, varname(this), this)
    # log_variable(namee, varname(nert_one), nert_one)
    # log_variable(namee, varname(again), again)
