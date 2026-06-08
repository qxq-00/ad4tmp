# Checkpoint download locations

## Required LDM checkpoint
Download:
https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

Place as:
`models/ldm/text2img-large/model.ckpt`

Absolute path on this machine:
`/Users/Zhuanz/Documents/New project/anomalydiffusion-deploy/anomalydiffusion/models/ldm/text2img-large/model.ckpt`

After downloading, verify with:

```bash
cd "/Users/Zhuanz/Documents/New project/anomalydiffusion-deploy/anomalydiffusion"
. .venv/bin/activate
python - <<'PY'
from pathlib import Path
p = Path('models/ldm/text2img-large/model.ckpt')
print(p.resolve())
print('exists:', p.exists(), 'size_gb:', round(p.stat().st_size/1024**3, 2) if p.exists() else None)
PY
```

## Optional/project checkpoints from README
- Generated data: https://drive.google.com/file/d/1fV2S-Memcll0oAnrPmfNLgi8E7yb7XTC/view?usp=drive_link
- Anomaly generation model: https://drive.google.com/drive/folders/17SA6QWGH4Mxk4lTIDm2DpG0N3PcpWicl?usp=sharing → `logs/anomaly-checkpoints`
- Mask generation model: https://drive.google.com/drive/folders/1LPJCd2dwocPHnA-Ex6d9aHFVk1JGHZ7Q?usp=sharing → `logs/mask-checkpoints`
- Localization checkpoints: https://drive.google.com/drive/folders/1PYq1I00JBij9J7IvNdYsQWLFnY0eQ20v?usp=sharing → `checkpoints/localization`
- Classification checkpoints: https://drive.google.com/drive/folders/1XhSaDZJQb9d6VYkf5GU3C8a4XgjGfB0N?usp=sharing → `checkpoints/classification`
