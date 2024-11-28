import kagglehub
import shutil
import os
from tqdm import tqdm

# Download latest version
download_path = kagglehub.dataset_download("secareanualin/football-events",target_path="DATA")


