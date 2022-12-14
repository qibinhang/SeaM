import sys
sys.path.append('..')
from global_config import GlobalConfig


def load_config():
    config = Config()
    return config


class Config(GlobalConfig):
    def __init__(self):
        super().__init__()
        self.project_data_save_dir = f'{self.data_dir}/defect_inheritance'
        self.standard_finetune_dir = f'{self.project_data_save_dir}/standard_ft'
        self.seam_finetune_dir = f'{self.project_data_save_dir}/seam_ft'
        self.retrain_dir = f'{self.project_data_save_dir}/retrain'