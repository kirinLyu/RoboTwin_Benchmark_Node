# RoboTwin G05 external-server runtime

The `g05_external` profile requires a prepared simulator host and an already
running G05 server. The Skill owns the Gateway, benchmark and policy client;
stopping it does not stop that external G05 server.

## Pinned sources and public simulator assets

- RoboTwin: https://github.com/RoboTwin-Platform/RoboTwin, commit
  `96c1feab536306b50c26af200044fcdf126e8904`.
- XPolicyLab (inside the RoboTwin checkout): https://github.com/XPolicyLab/XPolicyLab,
  commit `3f7823726fd0dd80705f009a94f59d9042ced56a`.
- Simulator assets: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0,
  revision `981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042`.

| Archive | SHA-256 |
| --- | --- |
| embodiments.zip | 6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6 |
| objects.zip | 6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27 |

Extract into RoboTwin's `assets/` and apply the upstream embodiment path
configuration. The supplied profile selects `demo_clean`, with random
backgrounds disabled; other scene configurations may need the additional
`background_texture.zip` asset. Prepare the upstream native motion planner
required by the selected embodiment, including cuRobo where configured.

The thin Node host must use Python 3.12 and NumPy 1.26.4. Preparation is being
validated with CUDA Torch 2.5.1, SAPIEN 3.0.0b1, mplib 0.2.1, toppra 0.6.10,
Open3D 0.19.0 and SciPy 1.14.1. Open3D 0.18 / SciPy 1.10 in the upstream default
recipe do not provide Python 3.12 wheels. Both sides of an accuracy comparison
must use the same validated simulator stack. The G05 model server uses its
own upstream environment and does not have to share the Node's Python version.

## Exact G05 checkpoint

https://huggingface.co/OpenGalaxea/G05/tree/main/g05-robotwin20

`g05-robotwin20/checkpoints/model_state_dict.pt` is 11,440,380,114 bytes, with SHA-256:

`300d24c691f8d76846a218b9bd670312cf577b701ea5863f221af1b8d1c0d637`

When gated, log in with an account that has accepted the model's conditions and
has access. Also obtain the matching Hydra config, dataset statistics, processor
files and action tokenizer expected by the pinned XPolicyLab G05 launcher.
Use the continuous FM head (`discrete_action=false`, `continuous_action=true`),
embodiment `robotwin`, joint actions of dimension 14, frequency 30 and action
steps 16. Use the same deploy settings for official and Forge evaluation.

## Launch environment

```bash
export ROBOTWIN_ROOT=/absolute/path/to/RoboTwin
export ROBOTWIN_PYTHON=/absolute/path/to/robotwin-venv/bin/python
export ROBOTWIN_ENV_SITE=/absolute/path/to/robotwin-venv/lib/python3.12/site-packages
export G05_SERVER_URL=ws://127.0.0.1:18091
export CUDA_VISIBLE_DEVICES=0
paos skill start robotwin-benchmark --profile g05_external
```

`start.sh` checks host dependencies and asset paths. The policy waits for the
XPolicyLab protocol handshake before registering `robotwin.policy.describe`.
This confirms the connection, while checkpoint identity and genuine inference
still need to be established from the server setup and real evaluation evidence.
Protocol tests using a fake server are not a G05 accuracy result.
