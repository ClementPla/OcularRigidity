import json
import uuid
from pathlib import Path


def cid():
    return uuid.uuid4().hex[:8]


def md(src):
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "id": cid(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


nb_path = Path(__file__).parent / "pixel_svd_diagnostics_viewer.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

nb["cells"].append(
    md(
        """## E. Video one-cycle

Repliement de `registered_frames` par phase (`NCycleReconstructor`, phase de
la section D) en `n_bins x n_cycle` frames moyennees/mediannees --
`one_cycle.mp4`, ecrite par `compute_one_cycle_pixel_svd.py` a cote du
fichier de diagnostics (meme dossier que `NPZ_PATH`)."""
    )
)

nb["cells"].append(
    code(
        """from IPython.display import Video

ONE_CYCLE_PATH = NPZ_PATH.parent / "one_cycle.mp4"
print(f"Video one-cycle : {ONE_CYCLE_PATH} ({ONE_CYCLE_PATH.stat().st_size / 1e6:.2f} Mo)")
Video(str(ONE_CYCLE_PATH), embed=True, html_attributes="controls loop", width=500)
"""
    )
)

nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("appended, n cells now:", len(nb["cells"]))
