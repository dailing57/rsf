
import yaml
with open('configs/semantic_cfg_dl.yaml') as file:
    cfg = yaml.safe_load(file)
    print(cfg['data']['pc_path'])
