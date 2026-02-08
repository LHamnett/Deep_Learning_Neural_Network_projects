import os
import yaml
from Bio.PDB import PDBParser

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
top_level_folder = os.path.join(root_dir,'deep_origin')

print(f'root dir: {root_dir}')

config_path = os.path.join(top_level_folder, 'src', 'config.yaml')

with open(config_path, 'r') as  yaml_file:
    config = yaml.safe_load(yaml_file)

if config is None:
    config = {}

config['root_dir'] = root_dir
config['top_level_folder'] = top_level_folder

with open(config_path, 'w') as yaml_file:
    yaml.dump(config, yaml_file)

# expects train and test data to be copied into data/raw folder
train_folder = os.path.join(top_level_folder,'data','raw','train')
train_pdb_files = os.listdir(train_folder)
test_file_location = os.path.join(train_folder,train_pdb_files[0])
print(f'test pdb filepath: {test_file_location}')

parser = PDBParser(QUIET=True)

try:
    
    structure = parser.get_structure("protein", test_file_location)

    # Print high-level structure info
    print(structure)
    print(f'num models: {len(structure)}')
    # for model in structure:
    #     print(f"Model ID: {model.id}")
    #     for chain in model:
    #         print(f"  Chain ID: {chain.id}, residues: {len(list(chain.get_residues()))}")
    print('pdb file loaded successfully')

except:
    raise EnvironmentError('Could not load pdb file for testing')

