# robotwin-benchmark 0.1.0 candidate

This bundle uses immutable Node locks from Ubuntu 20.04 builds. It is not accepted for publication until real-model and PAOS lifecycle tests pass.

Profiles: g05_external. Gateway: http://127.0.0.1:19132.

The g05_external profile attaches to an already running, user-configured XPolicyLab G05 websocket server. It owns only the client, never starts or stops that external server. G05 model setup is a prerequisite; an unavailable server must fail startup.

Required host environment: ROBOTWIN_ROOT, ROBOTWIN_ENV_SITE, ROBOTWIN_PYTHON, G05_SERVER_URL, CUDA_VISIBLE_DEVICES. Host runtime must be Python 3.12 with NumPy 1.26.4; select GPU(s) explicitly and ensure enough free memory. CALVIN uses the official MCIL checkpoint and L6 language embeddings. RoboTwin requires native simulation assets and the G05 robotwin20 model in its external policy server.

The host simulation/model packages are deliberately external to the thin Node binary. Environment preparation recipes and download/source hashes accompany the integration delivery. An existing interpreter alone is not proof that assets and imports are compatible.
