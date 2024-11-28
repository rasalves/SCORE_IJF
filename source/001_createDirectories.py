from pathlib import Path

dirs = [
    'DATA', 
    'RES', 
    'TRANSFER', 
    'YS', 
    'RES/MATCHES', 
    'RES/BASELINES', 
    'RES/INTERPRETABILITY', 
    'RES/FIGURES'
]

for dir_path in dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {dir_path}")