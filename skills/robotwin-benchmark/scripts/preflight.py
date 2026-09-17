"""Fail early on missing host prerequisites before starting the Skill's Nodes."""
from pathlib import Path
import hashlib
import importlib
import os
import sys
import sysconfig

def required(name):
    value=os.environ.get(name)
    if not value:raise RuntimeError(f'Missing environment variable: {name}')
    return value

def file(path):
    path=Path(path).expanduser()
    if not path.is_file():raise RuntimeError(f'Required file is missing: {path}')
    return path

def digest(path,expected):
    path=file(path)
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected:raise RuntimeError(f'SHA-256 mismatch: {path}; expected {expected}, got {actual}')

def main():
    benchmark=sys.argv[1]
    if sys.version_info[:2]!=(3,12):raise RuntimeError('The thin Nodes require a Python 3.12 host environment')
    required('CUDA_VISIBLE_DEVICES')
    import numpy as np
    if np.__version__!='1.26.4':raise RuntimeError('Host NumPy must match the Node ABI: 1.26.4')
    site_key='CALVIN_HOST_SITE' if benchmark=='calvin' else 'ROBOTWIN_ENV_SITE'
    selected=Path(required(site_key)).expanduser().resolve()
    if selected!=Path(sysconfig.get_path('purelib')).resolve():
        raise RuntimeError(f'{site_key} does not match the selected host interpreter')
    modules=('torch','hydra','pybullet','pytorch_lightning') if benchmark=='calvin' else ('torch','sapien','mplib','toppra','open3d','gymnasium')
    for name in modules:importlib.import_module(name)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in the selected host environment')
    if benchmark=='calvin':
        root=Path(required('CALVIN_ROOT')).expanduser()
        file(root/'calvin_models/calvin_agent/evaluation/evaluate_policy.py')
        file(root/'calvin_env/calvin_env/envs/play_table_env.py')
        data=Path(required('CALVIN_DATASET')).expanduser()
        file(data/'validation/.hydra/merged_config.yaml')
        file(Path(required('MCIL_TRAIN_FOLDER')).expanduser()/'.hydra/config.yaml')
        digest(required('MCIL_CHECKPOINT'),'cf2bc864900ce38b85daa68300addb8fdef79cc7102bd652a99f5f1924b69346')
        digest(data/'validation/lang_annotations/embeddings.npy','70f175572e2ad669eeb00d8a03e12ed796d3c55b2eb2c5c34eee9a8da8b5d25d')
    else:
        root=Path(required('ROBOTWIN_ROOT')).expanduser()
        file(root/'XPolicyLab/client_server/ws/model_client.py')
        file(root/'assets/objects/objaverse/list.json')
        if not (root/'assets/embodiments').is_dir():raise RuntimeError('RoboTwin embodiment assets are missing')
        if not required('G05_SERVER_URL').startswith(('ws://','wss://')):raise RuntimeError('G05_SERVER_URL must be a websocket URL')
    print(f'{benchmark} host preflight passed; model connection/loading is checked by policy readiness',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(f'Preflight failed: {error}',file=sys.stderr)
        raise SystemExit(1)
