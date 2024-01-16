import yaml

with open("./config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

MODEL_PATH  = cfg['MODEL_PATH']
CONFIG_PATH = cfg['CONFIG_PATH']