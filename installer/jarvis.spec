# PyInstaller build of the Jarvis-0 runtime (no training, no PyTorch). Used by build_installer.bat.
# Bundles your trained models from checkpoints/ (int8 copies when present) so a clean PC needs nothing else.
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
CK = ROOT / "checkpoints"
RUNTIME = ["*_encoder*.onnx", "*_decoder*.onnx", "*_onnx.json", "*tokenizer*.json", "ocr_*.onnx", "ocr_*.json",
           "meaning*.onnx", "meaning*.json"]
datas = []
for pattern in RUNTIME:
    for f in CK.glob(pattern):
        if ".tmp" not in f.name:
            datas.append((str(f), "checkpoints"))

hidden = [m for m in collect_submodules("jarvis") if ".tests" not in m] + [
    "training", "training.tokenizer", "training.loop", "training.ocr", "training.ocr.charset",
    "pystray._win32", "PIL.Image"]

a = Analysis([str(ROOT / "installer" / "jarvis_app.py")], pathex=[str(ROOT)], datas=datas, hiddenimports=hidden,
             excludes=["torch", "onnx", "pytest", "matplotlib", "IPython", "tensorboard", "training.train",
                       "training.model", "training.evaluate"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Jarvis", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Jarvis")
