import sys
# sys.path.append('rotation')
# sys.path.append('color')

import hydra
from omegaconf import OmegaConf

from color.flops_main_color import main_color
from rotation.main_rotation import main_rotation

@hydra.main(version_base=None, config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):
    # We want to add fields to cfg so need to call OmegaConf.set_struct
    OmegaConf.set_struct(cfg, False)
    print("Input arguments:")
    print(OmegaConf.to_yaml(cfg))

    if cfg.type == "color":
        main_color(cfg)
    elif cfg.type == "rotation":
        main_rotation(cfg)
    else:
        raise ValueError(f"Unknown task type: {cfg.type}")

if __name__ == "__main__":
    main()