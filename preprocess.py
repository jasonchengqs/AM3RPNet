import os
import sys
import joblib

import numpy as np
import pandas as pd
import argparse
import glob

import re
import stltovoxel

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    """Parser for command line arguments.
    """
    parser = argparse.ArgumentParser(description='Preprocessing configuration.')
    parser.add_argument('--mode', type=str, default='debug', 
                        help='run | debug')
    parser.add_argument('--gcode-dir', type=str, default='',
                        help='absolute path of gcode files.')
    parser.add_argument('--gcode-save-dir', type=str, default='',
                        help='absolute path to save processed gcode files.')
    parser.add_argument('--stl-dir', type=str, default='',
                        help='absolute path of stl files.')
    parser.add_argument('--stl-save-dir', type=str, default='',
                        help='absolute path to save processed stl files.')
    parser.add_argument('--layer-resource-dir', type=str, default='',
                        help='absolute path of layer energy files.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--save-slices', action='store_true',
                        help='choice to save binary slices from stl (default False).')                
    args = parser.parse_args()
    return args

# Preprocess GCODE files to timeseries files.
class GcodeProcessor():
    move_pattern = '^G1\s'
    layer_pattern = '^M118 NewLayer'

    def __init__(self, args):
        self.mode = args.mode
        self.files = sorted(glob.glob(os.path.join(args.gcode_dir, '*.gcode')))
        self.layer_resource_dir = args.layer_resource_dir
        if self.mode == 'debug':
            self.files = self.files[:3]
        self.save_dir = args.gcode_save_dir

    @staticmethod
    def parse_gcode(line):
        vars = {'F': None, 'X': None, 'Y': None, 'Z': None, 'E': None}
        new_layer = False
        if re.findall(GcodeProcessor.layer_pattern, line):
            new_layer = True
        elif re.findall(GcodeProcessor.move_pattern, line):
            for code in vars.keys():
                part = re.search('(\s' + code + ')([\d.-]+)(\s)', line)
                if part is not None:
                    vars[code] = float(part.groups()[1])
        return new_layer, vars

    def save(self, file_name, data):
        if not os.path.exists(os.path.join(self.save_dir, file_name)):
            os.makedirs(os.path.join(self.save_dir, file_name))
        for i, d in enumerate(data):
            joblib.dump(d, os.path.join(self.save_dir, file_name, f'timeseries_{file_name}_{i}.pkl'))
    
    def _measured_layers(self, file_name):
        df = pd.read_excel(os.path.join(self.layer_resource_dir, f'{file_name}_layer_data.xls'))
        return len(df)

    def process(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        for file in self.files:
            file_name = file.split('/')[-1].split('.gcode')[0]
            print(f'Processing {file}')
            file_vars = []
            layer = 0
            layer_vars = None
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    new_layer, line_vars = GcodeProcessor.parse_gcode(line)

                    # process new layer
                    if new_layer:
                        layer += 1
                        if layer > 1:
                            file_vars.append(layer_vars)
                        layer_vars = {'F': [], 'X': [], 'Y': [], 'Z': [], 'E': []}
                        
                    # save to current layer
                    elif layer > 0: # wait until printing of the 1st layer
                        for k, v in line_vars.items():
                            if v != None:
                                layer_vars[k].append(v)
                            elif len(layer_vars[k]) == 0: # no values yet
                                layer_vars[k].append(0.0)
                            else:
                                layer_vars[k].append(layer_vars[k][-1])
                    
                # save last layer
                file_vars.append(layer_vars)

            # check if #extracted layers are correct
            assert len(file_vars) == self._measured_layers(file_name), \
                f'Extracted {len(file_vars["F"])} != measured {self._measured_layers(file_name)}'
            
            self.save(file_name, file_vars)

# Preprocess STL files to voxel files.
class StlProcessor():
    def __init__(self, args):
        self.mode = args.mode
        self.files = sorted(glob.glob(os.path.join(args.stl_dir, '*.stl')))
        if self.mode == 'debug':
            self.files = self.files[:3]
        self.save_dir = args.stl_save_dir
        self.save_slices = args.save_slices
        self.layer_resource_dir = args.layer_resource_dir
        self.padding = 1 # padding at top and bottom after slicing

    def _measured_layers(self, file_name):
        df = pd.read_excel(os.path.join(self.layer_resource_dir, f'{file_name}_layer_data.xls'))
        return len(df)
    
    def process(self):
        if not os.path.exists(os.path.join(self.save_dir, 'coords')):
            os.makedirs(os.path.join(self.save_dir, 'coords'))
        
        for file in self.files:
            file_name = file.split('/')[-1].split('.stl')[0]
            print(f'Processing {file}')
            resolution = self._measured_layers(file_name)
            stltovoxel.convert_file(file, os.path.join(self.save_dir, 'coords', f'voxel_{file_name}.npy'), resolution=resolution, pad=self.padding)
            if self.save_slices:
                slice_save_dir = os.path.join(self.save_dir, 'slices', f'{file_name}')
                if not os.path.exists(slice_save_dir):
                    os.makedirs(slice_save_dir)
                stltovoxel.convert_file(file, os.path.join(slice_save_dir, f'voxel_{file_name}.png'), resolution=resolution, pad=self.padding)

if __name__ == '__main__':
    args = parse_args()

    # print('--- START PROCESSING GCODE FILES ---')
    # gp = GcodeProcessor(args)
    # gp.process()

    print('--- START PROCESSING STL FILES ---')
    sp = StlProcessor(args)
    sp.process()