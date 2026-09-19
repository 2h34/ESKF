"""Phase 3.8: read-only diagnosis of saved 15D results; no filter execution."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_eskf6d import segment_contiguous_indices


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def columns(rows, names):
    return np.array([[float(row[name]) for name in names] for row in rows])


def stats(values):
    return dict(zip(("mean", "median", "p95", "max"), map(float, (
        np.mean(values), np.median(values), np.percentile(values, 95), np.max(values)
    ))))


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze(result_dir, output_dir):
    result = read_csv(result_dir / "eskf15d_result.csv")
    debug_input = read_csv(result_dir / "eskf15d_debug.csv")
    source = json.loads((result_dir / "eskf15d_summary.json").read_text())
    indices = np.array([int(row["imu_index"]) for row in result])
    by_index = {int(row["imu_index"]): row for row in debug_input}
    if (len(by_index) != len(debug_input) or len(set(indices)) != len(result)
            or set(indices) != set(by_index)):
        raise ValueError("result/debug indices must be unique and identical")
    debug = [by_index[int(k)] for k in indices]
    timestamp = columns(result, ["timestamp"])[:, 0]
    t = timestamp - timestamp[0]
    if not np.all(np.diff(timestamp) > 0) or not np.all(np.diff(indices) == 1):
        raise ValueError("result time/index sequence is invalid")
    np.testing.assert_array_equal(timestamp, columns(debug, ["timestamp"])[:, 0])
    valid = np.array([row["update_applied"] == "True" for row in debug])
    if any(row["update_applied"] not in ("True", "False") for row in debug):
        raise ValueError("invalid update flag")
    if [row["update_applied"] for row in result] != [row["update_applied"] for row in debug]:
        raise ValueError("result/debug update flags disagree")
    if len(result) != source["total_output_frames"] or valid.sum() != source["successful_updates"]:
        raise ValueError("counts disagree with run summary")
    np.testing.assert_allclose(t[-1], source["duration"], rtol=0, atol=1e-9)
    residual_p = columns(debug, ["r_position_" + a for a in "xyz"])
    residual_q = columns(debug, ["r_theta_" + a for a in "xyz"])
    metrics = {name: columns(debug, [name])[:, 0] for name in (
        "NIS", "position_residual_norm", "attitude_residual_norm")}
    for name, values in metrics.items():
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
            raise ValueError(f"unexpected finite/missing data in {name}")
        np.testing.assert_allclose(list(stats(values[valid]).values()),
                                   [source[name][key] for key in stats(values[valid])],
                                   rtol=1e-12, atol=1e-14)
    for vector, name in ((residual_p, "position_residual_norm"),
                         (residual_q, "attitude_residual_norm")):
        np.testing.assert_allclose(np.linalg.norm(vector[valid], axis=1), metrics[name][valid],
                                   rtol=1e-12, atol=1e-14)

    p = columns(result, ["p" + a + "_m" for a in "xyz"])
    v = columns(result, ["v" + a + "_mps" for a in "xyz"])
    q = columns(result, ["qx", "qy", "qz", "qw"])
    rpy = columns(result, ["roll_rad", "pitch_rad", "yaw_rad"])
    omega = columns(debug, ["omega_hat_" + a for a in "xyz"])
    acc = columns(debug, ["a_hat_W_" + a for a in "xyz"])
    dt = columns(debug, ["dt"])[:, 0]
    for array in (p, v, q, rpy, omega[1:], acc[1:], dt[1:]):
        if not np.isfinite(array).all():
            raise ValueError("nonfinite state or propagation data")
    np.testing.assert_allclose(dt[1:], np.diff(timestamp), rtol=0, atol=1e-12)
    bookkeeping = {}
    for prefix, state, candidate in (
        ("delta_p_", p, p[:-1] + v[:-1] * dt[1:, None] + .5 * acc[1:] * dt[1:, None]**2),
        ("delta_v_", v, v[:-1] + acc[1:] * dt[1:, None]),
    ):
        correction = columns(debug, [prefix + a for a in "xyz"])
        if not np.isfinite(correction[valid]).all() or not np.isnan(correction[~valid]).all():
            raise ValueError("unexpected correction data")
        # Zero is used only in this algebraic check for a frame with no injection.
        applied = np.where(valid[:, None], correction, 0.0)
        error = float(np.max(np.abs(state[1:] - candidate - applied[1:])))
        bookkeeping[prefix + "max_recurrence_error"] = error
        if error > 1e-10:
            raise ValueError(f"saved state recurrence mismatch: {prefix}: {error}")

    motion = {
        "speed_mps": np.linalg.norm(v, axis=1),
        "omega_norm_rad_s": np.linalg.norm(omega, axis=1),
        "acc_norm_mps2": np.linalg.norm(acc, axis=1),
        "position_step_speed_mps": np.r_[np.nan, np.linalg.norm(np.diff(p, axis=0), axis=1) / dt[1:]],
        # acc[k] is the left-endpoint estimate at timestamp[k-1].
        "acc_change_rate_mps3": np.r_[np.nan, np.nan,
            np.linalg.norm(np.diff(acc[1:], axis=0), axis=1) / np.diff(timestamp[:-1])],
        "dt_s": dt,
    }
    nis = metrics["NIS"]
    threshold = float(np.percentile(nis[valid], 95))
    high = valid & (nis > threshold)
    high_rows = np.flatnonzero(high)

    def frame(i):
        return {"imu_index": int(indices[i]), "time_s": float(t[i]),
                **{key: float(value[i]) for key, value in metrics.items()},
                **{key: float(value[i]) if np.isfinite(value[i]) else None
                   for key, value in motion.items()}}

    segments = []
    for a, b in segment_contiguous_indices(indices[high]):
        rows = high_rows[a:b]
        segments.append({
            "start_index": int(indices[rows[0]]), "end_index": int(indices[rows[-1]]),
            "start_time_s": float(t[rows[0]]), "end_time_s": float(t[rows[-1]]),
            "duration_s": float(t[rows[-1]] - t[rows[0]]), "frame_count": len(rows),
            "max_NIS": float(nis[rows].max()),
            **{name + "_range": [float(values[rows].min()), float(values[rows].max())]
               for name, values in metrics.items() if name != "NIS"},
        })
    # Six at most: three longest and three highest-peak segments, without bridging gaps.
    selected = {s["start_index"]: s for s in
                sorted(segments, key=lambda s: s["frame_count"], reverse=True)[:3] +
                sorted(segments, key=lambda s: s["max_NIS"], reverse=True)[:3]}
    comparison = {}
    for name, mask in (("high", high), ("remaining_updates", valid & ~high)):
        comparison[name] = {"count": int(mask.sum()), **{
            key: stats(values[mask & np.isfinite(values)])
            for key, values in {**metrics, **motion}.items()}}
        comparison[name]["median_absolute_position_residual_xyz_m"] = np.median(abs(residual_p[mask]), axis=0).tolist()
        comparison[name]["median_absolute_attitude_residual_xyz_rad"] = np.median(abs(residual_q[mask]), axis=0).tolist()
    bins = []
    for start in np.arange(0, t[-1], 10):
        mask = valid & (t >= start) & (t < start + 10)
        bins.append({"start_s": float(start), "end_s": min(float(start+10), float(t[-1])),
                     "updates": int(mask.sum()), "high_count": int((mask & high).sum())})
    yaw_events = []
    for i in np.flatnonzero(abs(np.diff(rpy[:, 2])) > np.pi) + 1:
        angle = float(2 * np.arccos(np.clip(abs(q[i-1] @ q[i]), 0, 1)))
        yaw_events.append({"imu_index": int(indices[i]), "time_s": float(t[i]),
                           "yaw_before_rad": float(rpy[i-1, 2]), "yaw_after_rad": float(rpy[i, 2]),
                           "yaw_step_rad": float(rpy[i, 2]-rpy[i-1, 2]),
                           "shortest_rotation_rad": angle, "shortest_rotation_deg": float(np.rad2deg(angle))})
    six_path = result_dir / "eskf6d_debug.csv"
    six_comparison = None
    if six_path.exists():
        six = {int(r["imu_index"]): r for r in read_csv(six_path) if r["update_applied"] == "True"}
        common = [i for i in np.flatnonzero(valid) if int(indices[i]) in six]
        for i in common:
            if float(six[int(indices[i])]["timestamp"]) != timestamp[i]:
                raise ValueError("6D/15D joined timestamps disagree")
        a = np.array([float(six[int(indices[i])]["residual_norm"]) for i in common])
        b = metrics["attitude_residual_norm"][common]
        six_comparison = {"common_updates": len(common), "attitude_norm_correlation": correlation(a, b),
                          "max_absolute_norm_difference_rad": float(abs(a-b).max()),
                          "six_dimensional_NIS_compared": False,
                          "note": "6D filter observes 3D attitude; 15D observes 6D pose. NIS means do not rank accuracy."}
    summary = {
        "phase": "3.8", "output_frames": len(result), "update_count": int(valid.sum()),
        "excluded_nonupdate_count": int((~valid).sum()), "duration_s": float(t[-1]),
        "statistics": {key: stats(value[valid]) for key, value in metrics.items()},
        "source_summary_statistics_match": True, "saved_state_checks": bookkeeping,
        "maxima": {key: frame(int(np.nanargmax(value))) for key, value in metrics.items()},
        "high_NIS_selection": {"rule": "NIS > empirical p95 of valid updates; offline only, no gating",
                               "threshold": threshold, "frame_count": int(high.sum()),
                               "segment_count": len(segments), "ten_second_bins": bins},
        "representative_segments": sorted(selected.values(), key=lambda s: s["start_index"]),
        "high_vs_remaining": comparison,
        "correlations_log1p_NIS": {key: correlation(np.log1p(nis[valid & np.isfinite(value)]), value[valid & np.isfinite(value)])
                                   for key, value in {**motion, **{k:v for k,v in metrics.items() if k != "NIS"}}.items()},
        "large_dt_frames": [frame(i) for i in np.flatnonzero(dt > source["large_dt_threshold"])],
        "large_dt_threshold_s": source["large_dt_threshold"], "yaw_boundary_events": yaw_events,
        "six_dimensional_attitude_comparison": six_comparison,
        "interpretation_limits": [
            "Residual is prediction-to-observation discrepancy, not posterior discrepancy or ground-truth error.",
            "FAST-LIO consumes IMU and is not independent ground truth.",
            "Full innovation S is absent: NIS reconstruction and position/attitude percentage decomposition unavailable.",
            "Temporal correlation does not identify a unique physical cause or validate Q/R/P0.",
            "Chi-square(df=6) requires statistical assumptions not established here; no rejection criterion is used.",
            "Saved recurrence checks do not independently verify raw-IMU selection or pose matching.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(t, nis, linewidth=.7)
    ax.axhline(threshold, color="darkorange", linestyle="--", label="Empirical p95 (offline selection only)")
    ax.set(xlabel="Relative time (s)", ylabel="NIS (log scale)", title="15D ESKF Pose Innovation NIS", xlim=(0, t[-1]))
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "nis_time.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for row, (vector, norm, unit, title) in enumerate((
        (residual_p, metrics["position_residual_norm"], "m", "Position residual"),
        (residual_q, metrics["attitude_residual_norm"], "rad", "Attitude residual"))):
        for j, axis in enumerate("xyz"):
            axes[row, 0].plot(t, vector[:, j], label=axis, linewidth=.6, alpha=.8)
        axes[row, 0].legend(ncol=3)
        axes[row, 1].plot(t, norm, linewidth=.7)
        axes[row, 1].scatter(t[high], norm[high], s=3, color="darkorange", label="High NIS (empirical p95)")
        axes[row, 1].legend(fontsize=8)
        for col in range(2):
            axes[row, col].set(title=title + (" components" if col == 0 else " norm"), ylabel=unit, xlim=(0, t[-1]))
            axes[row, col].grid(alpha=.3)
    for ax in axes[-1]: ax.set_xlabel("Relative time (s)")
    fig.suptitle("15D ESKF Prediction-to-FAST-LIO Residuals")
    fig.tight_layout(); fig.savefig(output_dir / "residual_time.png", dpi=180); plt.close(fig)
    path = output_dir / "error_diagnosis_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis15d")
    args = parser.parse_args()
    analyze(args.result_dir, args.output_dir)
