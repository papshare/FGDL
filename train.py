import os
import sys
import shutil
import yaml
import argparse
from fgdl_model import FGDL
sys.path.append(".")
sys.path.append("..")

def load_config(file_path):
	with open(file_path, 'r') as file:
		return yaml.safe_load(file)
def main():
	parser = argparse.ArgumentParser(description="Accept config yaml file path")
	parser.add_argument('--config_file', type=str, default='ixi_cms.yaml')
	args = parser.parse_args()
	config = load_config(args.config_file)
	expn = config['exp_name']
	config['exp_dir'] = f'exps/{expn}'
	os.makedirs(config['exp_dir'], exist_ok=True)
	shutil.copy(args.config_file, f'{config["exp_dir"]}/config.yaml')
	fgdl = FGDL(config)
	fgdl.train()

if __name__ == '__main__':
	main()
