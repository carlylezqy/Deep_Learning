import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(f"DictConfig: {cfg}")
    print(f"[yaml]\n {OmegaConf.to_yaml(cfg)}")
    print(f"cfg.db.user: {cfg.db.user}")

if __name__ == "__main__":
    my_app()


"""
python 1.py db.user=root db.pass=1234
python 1.py +db.sceure_num=123
python 1.py ++db.pass=1234     #Existing key

"""