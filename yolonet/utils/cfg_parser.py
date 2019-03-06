import yaml
import sys
import logging as log
import os


def parser_yaml(filename):
    with open(filename,'r') as fp:
        config = yaml.load(fp.read())
        return config
def getConfig(cfg_root,model_name):
    main_cfg = parser_yaml('%s/main.yml' % cfg_root)
    if model_name not in main_cfg['cfg_dict'].keys():
        models = ', '.join(main_cfg['cfg_dict'].keys())
        print('There are models like %s\n' % models, file=sys.stderr)
        raise Exception
    cfg_fn = os.path.join(cfg_root,main_cfg['cfg_dict'][model_name])
    config =  parser_yaml(cfg_fn)
    return config