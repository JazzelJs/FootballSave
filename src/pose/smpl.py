"""Load the local SMPL body model used by the pose stage."""

from pathlib import Path
import pickle
import sys
import types

import numpy as np

MODEL = Path(__file__).parents[2] / "data/models/smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"


def load_model(model_path: Path = MODEL):
    """Return an SMPL model with the standard 24-joint skeleton."""
    import smplx
    from smplx.utils import Struct

    # The official Python download predates modern chumpy. Its only needed
    # chumpy value is shapedirs, which is just a NumPy-backed array.
    package = types.ModuleType("chumpy")
    package.__path__ = []
    module = types.ModuleType("chumpy.ch")

    class ChumpyArray:
        @property
        def shape(self):
            return self.x.shape

        def __array__(self, dtype=None):
            return np.asarray(self.x, dtype=dtype)

        def __getitem__(self, key):
            return self.x[key]

    module.Ch = ChumpyArray
    package.ch = ChumpyArray
    sys.modules.setdefault("chumpy", package)
    sys.modules.setdefault("chumpy.ch", module)

    with model_path.open("rb") as f:
        data = Struct(**pickle.load(f, encoding="latin1"))

    return smplx.SMPL(
        str(model_path),
        data_struct=data,
        gender="male",
        num_betas=10,
    )


if __name__ == "__main__":
    import torch

    model = load_model()
    assert model.J_regressor.shape == (24, 6890)
    output = model(
        betas=torch.zeros(1, 10),
        body_pose=torch.zeros(1, 69),
        global_orient=torch.zeros(1, 3),
    )
    assert output.joints.shape[1:] == (45, 3)
    assert output.joints[:, :24].shape[1:] == (24, 3)
    print(f"SMPL ok: {output.joints[:, :24].shape[1]} joints, {output.vertices.shape[1]} vertices")
