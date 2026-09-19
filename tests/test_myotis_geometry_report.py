import numpy as np

from acoustic_self_calibration.myotis import analyze_myotis_reference_geometry
from acoustic_self_calibration.simulation import (
    myotis_cross_microphones,
    nondegenerate_planar_microphones,
)


def test_geometry_report_flags_exact_cross_continuous_ambiguity() -> None:
    report = analyze_myotis_reference_geometry(myotis_cross_microphones())

    assert report["receiver_count"] == 12
    assert report["affine"]["rank"] == 2
    assert report["planar_conditioning"]["metric_design_rank"] == 4
    assert report["planar_conditioning"]["metric_nullity"] == 1
    assert report["planar_conditioning"]["quadratic_design_rank"] == 5
    assert report["planar_conditioning"]["conic_nullity"] == 1
    assert report["planar_conditioning"]["cross_like_degeneracy"] is True
    assert report["best_fit_plane"]["rms_normal_residual_m"] < 1e-12
    assert any("continuous" in item for item in report["ambiguities"])
    assert any("source sign" in item for item in report["ambiguities"])


def test_geometry_report_distinguishes_generic_planar_array() -> None:
    report = analyze_myotis_reference_geometry(nondegenerate_planar_microphones())

    assert report["affine"]["rank"] == 2
    assert report["planar_conditioning"]["metric_design_rank"] == 5
    assert report["planar_conditioning"]["metric_nullity"] == 0
    assert report["planar_conditioning"]["quadratic_design_rank"] == 6
    assert report["planar_conditioning"]["conic_nullity"] == 0
    assert report["planar_conditioning"]["cross_like_degeneracy"] is False
    assert report["best_fit_plane"]["rms_normal_residual_m"] < 1e-12
    assert (
        "generic planar Euclidean metric is locally determined" in report["identifiable_quantities"]
    )


def test_geometry_report_detects_small_out_of_plane_departure() -> None:
    microphones = myotis_cross_microphones().copy()
    microphones[:, 1] = np.linspace(-1e-3, 1e-3, len(microphones))
    report = analyze_myotis_reference_geometry(microphones)

    assert report["affine"]["rank"] == 3
    assert report["affine"]["s3_over_s1"] is not None
    assert report["best_fit_plane"]["rms_normal_residual_m"] > 0.0
