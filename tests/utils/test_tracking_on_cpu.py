import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from verl.utils.tracking import Tracking, _resolve_tracking_root


def test_resolve_tracking_root_uses_repo_outputs_and_experiment_basename(monkeypatch):
    monkeypatch.delenv("VERL_TRACKING_DIR", raising=False)

    tracking_root = _resolve_tracking_root("outputs/example_run")

    assert tracking_root == Path(__file__).resolve().parents[2] / "outputs" / "example_run"


def test_tracking_places_wandb_and_tensorboard_under_configured_root(tmp_path, monkeypatch):
    monkeypatch.delenv("VERL_TRACKING_DIR", raising=False)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.delenv("TENSORBOARD_DIR", raising=False)
    wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    config = {"trainer": {"tracking_dir": str(tmp_path)}}

    with patch("verl.utils.tracking._TensorboardAdapter") as tensorboard_adapter:
        tracking = Tracking(
            project_name="project",
            experiment_name="run",
            default_backend=["wandb", "tensorboard"],
            config=config,
        )

    assert wandb.init.call_args.kwargs["dir"] == str(tmp_path.resolve())
    tensorboard_adapter.assert_called_once_with("project", "run", tracking_root=tmp_path.resolve())
    assert Path(os.environ["TENSORBOARD_DIR"]) == tmp_path.resolve() / "tensorboard"
    tracking.__del__()
