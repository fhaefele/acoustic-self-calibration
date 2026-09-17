from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .evaluation import GroundTruthEvaluation, evaluate_against_ground_truth
from .ground_truth import GroundTruth
from .pipeline import AudioCalibrationResult


def _aligned_covariances(std_m: np.ndarray | None, rotation: np.ndarray) -> np.ndarray | None:
    if std_m is None:
        return None
    std = np.asarray(std_m, dtype=float)
    covariances = np.zeros((len(std), 3, 3), dtype=float)
    for index, row in enumerate(std):
        covariance = np.diag(row * row)
        covariances[index] = rotation.T @ covariance @ rotation
    return covariances


def _ellipse_from_covariance(
    center: np.ndarray,
    covariance: np.ndarray,
    dimensions: tuple[int, int],
) -> Ellipse:
    projected = covariance[np.ix_(dimensions, dimensions)]
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    width, height = 2.0 * np.sqrt(eigenvalues)
    return Ellipse(
        xy=(center[dimensions[0]], center[dimensions[1]]),
        width=float(width),
        height=float(height),
        angle=angle,
        fill=False,
        linewidth=0.7,
        alpha=0.22,
    )


def _plot_projection(
    axis: Any,
    estimate_mics: np.ndarray,
    estimate_source: np.ndarray,
    dimensions: tuple[int, int],
    labels: tuple[str, str],
    *,
    ground_truth_mics: np.ndarray | None,
    ground_truth_source: np.ndarray | None,
    mic_covariances: np.ndarray | None,
    source_covariances: np.ndarray | None,
) -> None:
    x, y = dimensions
    axis.scatter(
        estimate_mics[:, x],
        estimate_mics[:, y],
        marker="x",
        s=44,
        label="Estimated microphones",
    )
    axis.plot(
        estimate_source[:, x],
        estimate_source[:, y],
        linewidth=1.8,
        label="Estimated source",
    )

    if ground_truth_mics is not None and ground_truth_source is not None:
        axis.scatter(
            ground_truth_mics[:, x],
            ground_truth_mics[:, y],
            marker="o",
            facecolors="none",
            s=52,
            label="GT microphones",
        )
        axis.plot(
            ground_truth_source[:, x],
            ground_truth_source[:, y],
            linestyle="--",
            linewidth=1.6,
            label="GT source",
        )
        for estimate, truth in zip(estimate_mics, ground_truth_mics, strict=True):
            axis.plot(
                [estimate[x], truth[x]],
                [estimate[y], truth[y]],
                linewidth=0.7,
                alpha=0.35,
            )

    if mic_covariances is not None:
        for position, covariance in zip(estimate_mics, mic_covariances, strict=True):
            axis.add_patch(_ellipse_from_covariance(position, covariance, dimensions))
    if source_covariances is not None and len(estimate_source):
        stride = max(1, len(estimate_source) // 12)
        for index in range(0, len(estimate_source), stride):
            axis.add_patch(
                _ellipse_from_covariance(
                    estimate_source[index],
                    source_covariances[index],
                    dimensions,
                )
            )

    axis.set_xlabel(f"{labels[0]} [m]")
    axis.set_ylabel(f"{labels[1]} [m]")
    axis.set_title(f"{labels[0]}{labels[1]} projection")
    axis.axis("equal")
    axis.grid(True, alpha=0.25)


def plot_calibration_comparison(
    result: AudioCalibrationResult,
    path: str | Path,
    *,
    ground_truth: GroundTruth | None = None,
    evaluation: GroundTruthEvaluation | None = None,
) -> Path:
    """Write one 2x2 figure with 3-D, XY, XZ, and YZ views."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if ground_truth is not None:
        if evaluation is None:
            evaluation = evaluate_against_ground_truth(result, ground_truth)
        estimate_mics = evaluation.aligned_microphone_positions_m
        estimate_source = evaluation.aligned_source_positions_m
        truth_mics = ground_truth.microphone_positions_m
        truth_source = evaluation.ground_truth_source_at_estimate_times_m
        rotation = evaluation.rotation
    else:
        estimate_mics = result.calibration.microphone_positions
        estimate_source = result.calibration.source_positions
        truth_mics = None
        truth_source = None
        rotation = np.eye(3)

    mic_covariances = _aligned_covariances(
        result.calibration.microphone_position_std_m,
        rotation,
    )
    source_covariances = _aligned_covariances(
        result.calibration.source_position_std_m,
        rotation,
    )

    figure = plt.figure(figsize=(14, 11), constrained_layout=True)
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
    axis_xy = figure.add_subplot(2, 2, 2)
    axis_xz = figure.add_subplot(2, 2, 3)
    axis_yz = figure.add_subplot(2, 2, 4)

    axis_3d.scatter(
        estimate_mics[:, 0],
        estimate_mics[:, 1],
        estimate_mics[:, 2],
        marker="x",
        s=44,
        label="Estimated microphones",
    )
    axis_3d.plot(
        estimate_source[:, 0],
        estimate_source[:, 1],
        estimate_source[:, 2],
        linewidth=1.8,
        label="Estimated source",
    )
    if truth_mics is not None and truth_source is not None:
        axis_3d.scatter(
            truth_mics[:, 0],
            truth_mics[:, 1],
            truth_mics[:, 2],
            marker="o",
            facecolors="none",
            s=52,
            label="GT microphones",
        )
        axis_3d.plot(
            truth_source[:, 0],
            truth_source[:, 1],
            truth_source[:, 2],
            linestyle="--",
            linewidth=1.6,
            label="GT source",
        )
        for estimate, truth in zip(estimate_mics, truth_mics, strict=True):
            axis_3d.plot(
                [estimate[0], truth[0]],
                [estimate[1], truth[1]],
                [estimate[2], truth[2]],
                linewidth=0.7,
                alpha=0.35,
            )
    axis_3d.set_xlabel("X [m]")
    axis_3d.set_ylabel("Y [m]")
    axis_3d.set_zlabel("Z [m]")
    axis_3d.set_title("3-D scene")
    axis_3d.legend(loc="best", fontsize="small")
    axis_3d.set_box_aspect((1, 1, 1))

    _plot_projection(
        axis_xy,
        estimate_mics,
        estimate_source,
        (0, 1),
        ("X", "Y"),
        ground_truth_mics=truth_mics,
        ground_truth_source=truth_source,
        mic_covariances=mic_covariances,
        source_covariances=source_covariances,
    )
    _plot_projection(
        axis_xz,
        estimate_mics,
        estimate_source,
        (0, 2),
        ("X", "Z"),
        ground_truth_mics=truth_mics,
        ground_truth_source=truth_source,
        mic_covariances=mic_covariances,
        source_covariances=source_covariances,
    )
    _plot_projection(
        axis_yz,
        estimate_mics,
        estimate_source,
        (1, 2),
        ("Y", "Z"),
        ground_truth_mics=truth_mics,
        ground_truth_source=truth_source,
        mic_covariances=mic_covariances,
        source_covariances=source_covariances,
    )

    if evaluation is not None:
        figure.suptitle(
            "Ground-truth comparison — "
            f"mic RMS {evaluation.microphone_rms_error_m:.4f} m, "
            f"source RMS {evaluation.source_rms_error_m:.4f} m"
        )
    else:
        figure.suptitle("Acoustic self-calibration result")

    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target
