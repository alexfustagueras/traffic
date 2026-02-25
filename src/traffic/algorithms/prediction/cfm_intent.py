#!/usr/bin/env python3
"""
Intent-conditioned CFM prediction for aircraft trajectories

This predictor extends the existing CFM inference path by augmenting the
context with intent features computed from the 60 s history:
  - 12 continuous kinematic intent features
  - 25 one-hot discrete intent label

The loaded checkpoint must be trained with context_dim=45
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ...core.flight import Flight
from ...core.traffic import Traffic
from . import PredictorBase
from .cfm import (
    aircraft_centric_transform,
    denorm_seq_to_global,
    load_model_checkpoint,
    sample_future_heun,
)

# Feature indices in 7-D state vector
IDX_VZ = 5
IDX_PSI = 6

# Intent thresholds
VZ_LEVEL_THR = 0.508
VZ_STRONG_THR = 2.54
PSI_STRAIGHT_THR = 0.005
PSI_TURN_THR = 0.01

# Intent windows (in samples at 1 Hz)
NOW_WINDOW = 5
PRIOR_WINDOW = 15
EARLY_WINDOW = 15
TREND_WINDOW = 15


def _linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    xm = x.mean()
    ym = y.mean()
    den = ((x - xm) ** 2).sum()
    if den < 1e-12:
        return 0.0
    num = ((x - xm) * (y - ym)).sum()
    return float(num / den)


def compute_intent_features(hist_phys: np.ndarray) -> np.ndarray:
    """Compute the 12 continuous intent features from denormalized history."""
    T = hist_phys.shape[0]
    vz = hist_phys[:, IDX_VZ]
    psi = hist_phys[:, IDX_PSI]

    late_sl = slice(max(0, T - TREND_WINDOW), T)
    early_sl = slice(0, min(EARLY_WINDOW, T))

    mean_vz_late = np.mean(vz[late_sl])
    mean_vz_early = np.mean(vz[early_sl])
    mean_psi_late = np.mean(psi[late_sl])
    mean_psi_early = np.mean(psi[early_sl])

    t_axis = np.arange(T, dtype=np.float64)
    dvz_dt = _linreg_slope(t_axis, vz.astype(np.float64))
    dpsi_dt = _linreg_slope(t_axis, psi.astype(np.float64))

    return np.array(
        [
            mean_vz_late,
            mean_vz_early,
            dvz_dt,
            vz[-1],
            mean_psi_late,
            mean_psi_early,
            dpsi_dt,
            psi[-1],
            np.abs(mean_vz_late),
            np.abs(mean_psi_late),
            mean_vz_late - mean_vz_early,
            mean_psi_late - mean_psi_early,
        ],
        dtype=np.float32,
    )


def classify_intent_index(hist_phys: np.ndarray) -> int:
    """Classify history into flat intent class index in [0, 25)."""
    vz = hist_phys[:, IDX_VZ]
    psi = hist_phys[:, IDX_PSI]
    T = len(vz)

    now_start = max(0, T - NOW_WINDOW)
    prior_start = max(0, now_start - PRIOR_WINDOW)

    vz_now = float(np.mean(vz[now_start:]))
    psi_now = float(np.mean(psi[now_start:]))
    vz_prior = float(np.mean(vz[prior_start:now_start])) if now_start > prior_start else 0.0
    psi_prior = float(np.mean(psi[prior_start:now_start])) if now_start > prior_start else 0.0

    # Vertical class [0..4]
    if np.abs(vz_now) > VZ_LEVEL_THR:
        v = 1 if vz_now > 0 else 2
    elif np.abs(vz_prior) > VZ_STRONG_THR:
        v = 3 if vz_prior > 0 else 4
    else:
        v = 0

    # Lateral class [0..4]
    if np.abs(psi_now) > PSI_STRAIGHT_THR:
        l_ = 2 if psi_now > 0 else 1
    elif np.abs(psi_prior) > PSI_TURN_THR:
        l_ = 4 if psi_prior > 0 else 3
    else:
        l_ = 0

    return int(v * 5 + l_)


class CFMIntentPredict(PredictorBase):
    """Intent-conditioned CFM predictor using a single model checkpoint."""

    method_name = "cfm_intent"

    def __init__(
        self,
        model_path: str,
        stats_path: Optional[str] = None,
        intent_norm_path: Optional[str] = None,
        device: Optional[str] = None,
        n_samples: int = 10,
        n_steps: int = 64,
        guidance_scale: float = 1.0,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = load_model_checkpoint(model_path, self.device)
        ckpt = torch.load(model_path, map_location="cpu")
        checkpoint_norm_stats = ckpt.get("norm_stats")
        checkpoint_intent_norm = ckpt.get("intent_norm_stats")

        if stats_path is not None:
            with open(stats_path, "r") as f:
                self.norm_stats = json.load(f)
        elif checkpoint_norm_stats is not None:
            self.norm_stats = checkpoint_norm_stats
        else:
            raise ValueError(
                "stats_path is required unless normalization stats are embedded "
                "in the checkpoint under key 'norm_stats'"
            )

        # Intent-feature normalization stats
        if intent_norm_path is None and checkpoint_intent_norm is None:
            sidecar = Path(model_path).with_suffix(".intent_norm.json")
            if sidecar.exists():
                intent_norm_path = str(sidecar)

        if intent_norm_path is None and checkpoint_intent_norm is None:
            raise ValueError(
                "intent_norm_path is required (or provide "
                "<model>.intent_norm.json sidecar, or embed 'intent_norm_stats' in checkpoint)"
            )

        if intent_norm_path is not None:
            with open(intent_norm_path, "r") as f:
                intent_norm = json.load(f)
        else:
            intent_norm = checkpoint_intent_norm

        self.n_samples = n_samples
        self.n_steps = n_steps
        self.guidance_scale = guidance_scale

        self.feat_mean = np.array(self.norm_stats["feat_mean"], dtype=np.float32)
        self.feat_std = np.array(self.norm_stats["feat_std"], dtype=np.float32)
        self.ctx_mean = np.array(self.norm_stats["ctx_mean"], dtype=np.float32)
        self.ctx_std = np.array(self.norm_stats["ctx_std"], dtype=np.float32)

        self.intent_feat_mean = np.array(
            intent_norm["intent_feat_mean"], dtype=np.float32
        )
        self.intent_feat_std = np.array(
            intent_norm["intent_feat_std"], dtype=np.float32
        )

    def _resample_predictions(self, predictions: torch.Tensor, sampling_rate: float) -> torch.Tensor:
        """Resample +5,+10,...,+60 s predictions to a denser sampling rate."""
        assert predictions.dim() == 3, "predictions must be (S, T, D)"
        S, _T, D = predictions.shape
        original_times = np.arange(5, 61, 5, dtype=float)
        new_times = np.arange(5, 60 + 1e-6, sampling_rate, dtype=float)
        out = torch.empty((S, len(new_times), D), dtype=predictions.dtype, device=predictions.device)
        pred_np = predictions.detach().cpu().numpy()
        for f in range(D):
            vals = pred_np[:, :, f]
            res = np.vstack(
                [np.interp(new_times, original_times, vals_i) for vals_i in vals]
            )
            out[:, :, f] = torch.from_numpy(res).to(out.device, out.dtype)
        return out

    def preprocess_flight(self, flight: Flight) -> tuple[np.ndarray, pd.Timestamp]:
        """Convert a Flight object into 60x7 physical feature matrix."""
        window = flight.last(seconds=60)
        if window is None or len(window.data) < 60:
            raise ValueError("Flight must have at least 60 seconds of data")

        df = window.data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if len(df) != 60:
            raise ValueError(f"Expected exactly 60 samples, got {len(df)}")
        last_timestamp = df["timestamp"].iloc[-1]

        import pyproj

        to_lv95 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
        x_coords, y_coords = to_lv95.transform(
            df["longitude"].to_numpy(), df["latitude"].to_numpy()
        )
        df["x"] = x_coords
        df["y"] = y_coords

        knots_to_mps = 0.5144444444444444
        ftpm_to_mps = 0.3048 / 60.0
        track_rad = np.deg2rad(df["track"].to_numpy())
        spd_mps = df["groundspeed"].to_numpy() * knots_to_mps
        df["vx"] = spd_mps * np.sin(track_rad)
        df["vy"] = spd_mps * np.cos(track_rad)
        df["z"] = df["altitude"] * 0.3048
        df["vz"] = df["vertical_rate"] * ftpm_to_mps

        vx = df["vx"].to_numpy()
        vy = df["vy"].to_numpy()
        vxm = np.roll(vx, 1)
        vxm[0] = vx[0]
        vym = np.roll(vy, 1)
        vym[0] = vy[0]
        cross = vxm * vy - vym * vx
        dot = vxm * vx + vym * vy
        psi_rate = -np.arctan2(cross, dot)
        df["psi_rate"] = np.clip(
            np.nan_to_num(psi_rate, nan=0.0, posinf=0.0, neginf=0.0),
            -0.25,
            0.25,
        )

        features = ["x", "y", "z", "vx", "vy", "vz", "psi_rate"]
        X_raw = df[features].to_numpy().astype(np.float32)
        return X_raw, last_timestamp

    def predict(self, flight: Flight) -> Flight | Traffic:
        X_raw, last_timestamp = self.preprocess_flight(flight)

        # Aircraft-centric transform and base context
        X_raw_b = X_raw[None, :, :]
        Y_dummy = np.zeros((1, 1, X_raw.shape[1]), dtype=X_raw.dtype)
        X_t_b, _, C_raw_b = aircraft_centric_transform(X_raw_b, Y_dummy)
        X_t, C_raw = X_t_b[0], C_raw_b[0]

        X_norm = ((X_t - self.feat_mean) / self.feat_std).astype(np.float32)
        C_norm_8 = (
            (C_raw - self.ctx_mean[: len(C_raw)]) / self.ctx_std[: len(C_raw)]
        ).astype(np.float32)

        # Build intent-extended context [8 + 12 + 25]
        X_phys = X_norm * self.feat_std[None, :] + self.feat_mean[None, :]
        intent_feats = compute_intent_features(X_phys)
        intent_feats_norm = (
            (intent_feats - self.intent_feat_mean) / self.intent_feat_std
        ).astype(np.float32)
        intent_idx = classify_intent_index(X_phys)
        intent_onehot = np.zeros(25, dtype=np.float32)
        intent_onehot[intent_idx] = 1.0
        C_norm_45 = np.concatenate(
            [C_norm_8, intent_feats_norm, intent_onehot], axis=0
        ).astype(np.float32)

        x_hist = torch.from_numpy(X_norm).unsqueeze(0).to(self.device)
        ctx_8 = torch.from_numpy(C_norm_8).unsqueeze(0).to(self.device)
        ctx_45 = torch.from_numpy(C_norm_45).unsqueeze(0).to(self.device)

        repeated_hist = x_hist.repeat(self.n_samples, 1, 1)
        repeated_ctx45 = ctx_45.repeat(self.n_samples, 1)
        T_out = 12
        futures_norm = sample_future_heun(
            self.model,
            repeated_hist,
            repeated_ctx45,
            T_out=T_out,
            n_steps=self.n_steps,
            G=self.guidance_scale,
        )

        # Use original 8-D context for denormalization/global conversion.
        futures_global = denorm_seq_to_global(
            futures_norm,
            ctx_8.repeat(self.n_samples, 1),
            self.feat_mean,
            self.feat_std,
            self.ctx_mean,
            self.ctx_std,
        )
        futures_global = self._resample_predictions(futures_global, sampling_rate=1)

        futures_np = futures_global.detach().cpu().numpy()
        import pyproj

        to_wgs84 = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

        predicted_flights = []
        for i in range(self.n_samples):
            lon, lat = to_wgs84.transform(futures_np[i, :, 0], futures_np[i, :, 1])
            alt_ft = futures_np[i, :, 2] / 0.3048
            # Predictions correspond to +5..+60s horizon and are resampled at 1s.
            pred_timestamps = [
                last_timestamp + timedelta(seconds=5 + j)
                for j in range(len(lon))
            ]
            flight_id = flight.identifier if hasattr(flight, "identifier") else "predicted"
            if self.n_samples > 1:
                flight_id = f"{flight_id}_sample_{i}"
            pred_data = pd.DataFrame(
                {
                    "timestamp": pred_timestamps,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt_ft,
                    "flight_id": flight_id,
                }
            )
            predicted_flights.append(Flight(pred_data))

        if self.n_samples == 1:
            return predicted_flights[0]
        return Traffic.from_flights(predicted_flights)