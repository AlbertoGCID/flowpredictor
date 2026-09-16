"""Load and validate ``experiments_config.toml`` into frozen dataclasses.

Single source of truth for the paths, cross-validation settings,
hyperparameters, schema columns and ablation variants used by
``main_pipeline.py``, ``run_experiments_phase3.py``,
``data/dataset_generator.py``, ``pipeline/iteration.py`` and
``visualization/*.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple

import tomli  # tomllib is only in the stdlib from Python 3.11; tomli has the
              # same API, so migrating is a one-line import change.


@dataclass(frozen=True)
class AblationVariant:
    """One ablation configuration (``[ablation_variants.<name>]``)."""
    description: str
    use_pretrain: bool
    use_rf: bool
    loss_name: str
    penalty: Tuple[int, ...]  # tuple, not list: truly immutable
    encoder_units: int
    decoder_units: int


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration, one read-only mapping per TOML section."""
    paths: Mapping[str, str]
    meta: Mapping[str, object]
    cv: Mapping[str, object]
    holdout: Mapping[str, object]
    base: Mapping[str, object]
    schema: Mapping[str, object]
    robustness: Mapping[str, object]
    dataset: Mapping[str, object]
    reporting: Mapping[str, object]
    variants: Mapping[str, AblationVariant]

    @classmethod
    def from_toml(cls, path: Path) -> "PipelineConfig":
        """Parse the TOML file and resolve absolute paths.

        Args:
            path (Path): Path to ``experiments_config.toml``.

        Returns:
            PipelineConfig: The frozen configuration.
        """
        with open(path, "rb") as f:
            raw = tomli.load(f)

        # Anchor for absolute paths: <path> = .../flowpredictor/resultados/prueba63/
        # experiments_config.toml -> parents[1] = .../flowpredictor. Computed here,
        # in an explicit function, never as a side effect of importing a module.
        prueba_root = path.resolve().parent
        project_root = prueba_root.parents[1]

        raw_paths = raw["paths"]
        doc_dir = prueba_root / raw_paths["doc_dir_name"]
        paths = {
            "result_path": str(project_root / raw_paths["result_path_rel"]),
            "data_bundle_root": str(project_root / raw_paths["data_bundle_root_rel"]),
            "results_csv": str(prueba_root / raw_paths["results_csv_name"]),
            "predictions_dir": str(prueba_root / raw_paths["predictions_dir_name"]),
            "doc_dir": str(doc_dir),
            "cache_root": str(prueba_root / raw_paths["cache_root_name"]),
            "yearly_plots_dir": str(doc_dir / "yearly_plots"),
        }

        variants = {
            name: AblationVariant(
                description=v["description"],
                use_pretrain=v["use_pretrain"],
                use_rf=v["use_rf"],
                loss_name=v["loss_name"],
                penalty=tuple(v["penalty"]),
                encoder_units=v["encoder_units"],
                decoder_units=v["decoder_units"],
            )
            for name, v in raw["ablation_variants"].items()
        }

        return cls(
            paths=MappingProxyType(paths),
            meta=MappingProxyType(dict(raw["meta"])),
            cv=MappingProxyType(dict(raw["cv"])),
            holdout=MappingProxyType(dict(raw["holdout"])),
            base=MappingProxyType(dict(raw["base_hyperparameters"])),
            schema=MappingProxyType(dict(raw["schema"])),
            robustness=MappingProxyType(dict(raw["robustness"])),
            dataset=MappingProxyType(dict(raw["dataset"])),
            reporting=MappingProxyType(dict(raw["reporting"])),
            variants=MappingProxyType(variants),
        )


_TOML_PATH = Path(__file__).resolve().parents[1] / "experiments_config.toml"
CONFIG = PipelineConfig.from_toml(_TOML_PATH)
