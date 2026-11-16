"""Backport the one-line upstream AutoModel0.7 Triton compatibility fix.
Source: NVIDIA-NeMo/Automodel c57935fc64faea913ce9f5e38ea7e57e6416cc93,
components/_peft/lora_kernel.py: lora_db_kernel autotune key excludes nonexistent S.
Only use inside the dedicated tutorial environment; no CUDA kernels are changed.
"""
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
if importlib.metadata.version('nemo-automodel') != '0.7.0':
    raise RuntimeError('This compatibility patch is only for release0.7.0')
p=Path(importlib.util.find_spec('nemo_automodel').origin).parent/'components/_peft/lora_kernel.py'
text=p.read_text()
old='key=["S", "M", "K"],'
new='key=["M", "K"],'
before=hashlib.sha256(text.encode()).hexdigest()
if old in text:
    if text.count(old)!=1: raise RuntimeError('Unexpected source; refusing patch')
    backup=p.with_suffix('.py.original-0.7')
    if not backup.exists(): backup.write_text(text)
    text=text.replace(old,new)
    p.write_text(text)
    state='applied'
elif new in text:
    state='already_fixed'
else:
    raise RuntimeError('Unexpected source; refusing patch')
print(json.dumps({'patch':state,'upstream_commit':'c57935fc64faea913ce9f5e38ea7e57e6416cc93','before_sha256':before,'after_sha256':hashlib.sha256(text.encode()).hexdigest()}))
