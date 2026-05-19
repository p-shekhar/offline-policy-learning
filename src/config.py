from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem contract for the support-aware offline policy learning project."""

    project_root: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "ProjectPaths":
        return cls(project_root=Path(project_root).expanduser().resolve())

    @classmethod
    def from_repo_root(cls, repo_root: str | Path) -> "ProjectPaths":
        """Build paths when the GitHub repository root is the code folder itself."""

        return cls.from_project_root(repo_root)

    @classmethod
    def infer_from_notebook(cls) -> "ProjectPaths":
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
                return cls(candidate)
            nested_code = candidate / "code"
            if (nested_code / "src").is_dir() and (nested_code / "notebooks").is_dir():
                return cls(candidate)
        return cls(cwd)

    @property
    def code_dir(self) -> Path:
        if (self.project_root / "src").is_dir():
            return self.project_root
        return self.project_root / "code"

    @property
    def data_dir(self) -> Path:
        repo_data = self.project_root / "data"
        candidates = [self.project_root, self.code_dir.parent, *self.project_root.parents]
        seen: set[Path] = set()
        for root in candidates:
            data = root / "data"
            if data in seen:
                continue
            seen.add(data)
            if data.exists():
                return data
        return repo_data

    @property
    def artifact_dir(self) -> Path:
        return self.code_dir / "artifacts"

    @property
    def metadata_dir(self) -> Path:
        return self.artifact_dir / "metadata"

    @property
    def panel_dir(self) -> Path:
        return self.artifact_dir / "panels"

    @property
    def table_dir(self) -> Path:
        return self.artifact_dir / "tables"

    @property
    def figure_dir(self) -> Path:
        return self.artifact_dir / "figures"

    @property
    def ipinyou_archive(self) -> Path:
        return self.data_dir / "archive.zip"

    def ensure(self) -> None:
        for folder in [
            self.artifact_dir,
            self.metadata_dir,
            self.panel_dir,
            self.table_dir,
            self.figure_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    """Common knobs used by notebooks.

    Set ``full_run=True`` in the notebooks to process all available rows. Keeping
    the knob explicit makes draft/debug runs reproducible without changing code.
    """

    full_run: bool = True
    quick_rows_per_day: int = 150_000
    random_seed: int = 20260513
    value_proxy_conversion_weight: float = 10.0
    lower_bound_alpha: float = 0.05
    support_penalty_scale: float = 1.0
    support_radius: float = 10.0
    shortlist_tolerance: float = 0.0
    min_segment_observations: int = 1_000
    bootstrap_iterations: int = 1_000
