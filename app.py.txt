import time
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from channel_model import transmit_ook_frames
from models import (
    UnfoldedFSODetNet,
    JointDNN,
    ls_zf_detector,
    genie_ml_detector,
    train_quick_models,
)

st.set_page_config(
    page_title="8×8 Massive MIMO-FSO | Unfolded FSO-DetNet",
    layout="wide",
)

st.title("Deep Learning-Based Joint Channel Estimation & Symbol Detection")
st.markdown(
    "**8×8 Massive MIMO-FSO Communication System** | Model-Driven **Unfolded FSO-DetNet** vs. Conventional Joint DNN, LS-ZF & Genie-Aided ML"
)


@st.cache_resource
def get_trained_models():
    """Trains and caches the Unfolded FSO-DetNet and Joint DNN models once."""
    return train_quick_models(transmit_ook_frames, steps=240, batch_size=128)


with st.spinner("Initializing & calibrating Unfolded FSO-DetNet (6-layer PGD) and Joint DNN..."):
    detnet_model, joint_dnn_model = get_trained_models()

# --- Sidebar Controls ---
st.sidebar.header("FSO Channel & System Settings")
turbulence_regime = st.sidebar.selectbox(
    "Atmospheric Turbulence (Gamma-Gamma)",
    ["Weak", "Moderate", "Strong"],
    index=1,
)
spatial_rho = st.sidebar.slider(
    "Spatial Correlation (Gaussian Copula ρ)",
    min_value=0.0,
    max_value=0.9,
    value=0.5,
    step=0.1,
)
selected_snr = st.sidebar.slider(
    "Operating Point SNR (dB)",
    min_value=5,
    max_value=25,
    value=15,
    step=5,
)
frames_per_snr = st.sidebar.slider(
    "Test Frames per SNR Point",
    min_value=100,
    max_value=1000,
    value=300,
    step=100,
)
training_frames_display = st.sidebar.number_input(
    "Offline Training Dataset Size (Frames)",
    value=40000,
    disabled=True,
)

run_sim = st.sidebar.button("Run Full 5–25 dB Evaluation", type="primary")


def compute_nmse_db(H_est, H_true):
    num = np.sum((H_est - H_true) ** 2)
    den = np.sum(H_true ** 2) + 1e-12
    return 10.0 * np.log10(num / den + 1e-12)


def compute_ber(X_pred_soft, X_true):
    X_bin = (X_pred_soft >= 0.5).astype(np.float32)
    return float(np.mean(X_bin != X_true))


# Run evaluation automatically on first load or when button is clicked
snr_range = [5, 10, 15, 20, 25]

results = {
    "snr": snr_range,
    "ber_detnet": [],
    "ber_dnn": [],
    "ber_lszf": [],
    "ber_ml": [],
    "nmse_detnet": [],
    "nmse_dnn": [],
    "nmse_ls": [],
    "lat_detnet": [],
    "lat_dnn": [],
    "lat_lszf": [],
    "lat_ml": [],
}

snapshot = {}

with st.spinner(f"Simulating 8×8 MIMO-FSO frames ({turbulence_regime} turbulence, ρ = {spatial_rho})..."):
    for snr in snr_range:
        H_true, P, Y_pilot, Y_data, X_true = transmit_ook_frames(
            num_frames=frames_per_snr,
            snr_db=float(snr),
            regime=turbulence_regime,
            rho=spatial_rho,
        )

        Yp_t = torch.tensor(Y_pilot)
        Yd_t = torch.tensor(Y_data)
        P_t = torch.tensor(P)

        # 1. Proposed Unfolded FSO-DetNet
        t0 = time.perf_counter()
        with torch.no_grad():
            H_det, X_det = detnet_model(Yp_t, Yd_t, P_t)
        t_det = (time.perf_counter() - t0) * 1000.0 / frames_per_snr
        H_det_np = H_det.numpy()
        X_det_np = X_det.numpy()

        # 2. Conventional Joint DNN
        t0 = time.perf_counter()
        with torch.no_grad():
            H_dnn, X_dnn = joint_dnn_model(Yp_t, Yd_t)
        t_dnn = (time.perf_counter() - t0) * 1000.0 / frames_per_snr
        H_dnn_np = H_dnn.numpy()
        X_dnn_np = X_dnn.numpy()

        # 3. Conventional LS + Regularized ZF
        t0 = time.perf_counter()
        H_ls_np, X_lszf_np = ls_zf_detector(Y_pilot, Y_data, P)
        t_lszf = (time.perf_counter() - t0) * 1000.0 / frames_per_snr

        # 4. Genie-Aided ML (Perfect CSI)
        t0 = time.perf_counter()
        X_ml_np = genie_ml_detector(H_true, Y_data)
        t_ml = (time.perf_counter() - t0) * 1000.0 / frames_per_snr

        results["ber_detnet"].append(max(compute_ber(X_det_np, X_true), 1e-5))
        results["ber_dnn"].append(max(compute_ber(X_dnn_np, X_true), 1e-5))
        results["ber_lszf"].append(max(compute_ber(X_lszf_np, X_true), 1e-5))
        results["ber_ml"].append(max(compute_ber(X_ml_np, X_true), 1e-5))

        results["nmse_detnet"].append(compute_nmse_db(H_det_np, H_true))
        results["nmse_dnn"].append(compute_nmse_db(H_dnn_np, H_true))
        results["nmse_ls"].append(compute_nmse_db(H_ls_np, H_true))

        results["lat_detnet"].append(t_det)
        results["lat_dnn"].append(t_dnn)
        results["lat_lszf"].append(t_lszf)
        results["lat_ml"].append(t_ml)

        if snr == selected_snr:
            snapshot = {
                "H_true": H_true[0],
                "H_det": H_det_np[0],
                "H_ls": H_ls_np[0],
                "ber_det": results["ber_detnet"][-1],
                "ber_lszf": results["ber_lszf"][-1],
                "nmse_det": results["nmse_detnet"][-1],
                "nmse_ls": results["nmse_ls"][-1],
                "lat_det": t_det,
                "lat_ml": t_ml,
            }

# --- Top KPI Row ---
st.subheader(f"Performance Summary at SNR = {selected_snr} dB ({turbulence_regime} Turbulence)")
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "FSO-DetNet BER",
    f"{snapshot['ber_det']:.2e}",
    delta=f"vs LS-ZF: {snapshot['ber_lszf']:.2e}",
    delta_color="inverse",
)
k2.metric(
    "Channel NMSE (dB)",
    f"{snapshot['nmse_det']:.2f} dB",
    delta=f"{snapshot['nmse_det'] - snapshot['nmse_ls']:.2f} dB vs LS",
    delta_color="inverse",
)
k3.metric(
    "FSO-DetNet Latency",
    f"{snapshot['lat_det']:.4f} ms/frame",
    delta=f"{snapshot['lat_ml'] / max(snapshot['lat_det'], 1e-6):.1f}× faster than ML",
)
k4.metric(
    "System Architecture",
    "8×8 MIMO-FSO",
    delta="IM/DD OOK | 6-Layer PGD",
)

st.divider()

# --- Tabs for IEEE Paper Figures & Channel Inspection ---
tab1, tab2, tab3 = st.tabs([
    "BER & Channel NMSE Curves",
    "Inference Latency Comparison",
    "8×8 Optical Channel Matrix Inspection",
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.2))
        ax1.semilogy(snr_range, results["ber_lszf"], "o--", label="LS-ZF (Conventional)", linewidth=2)
        ax1.semilogy(snr_range, results["ber_dnn"], "s-.", label="Joint DNN (Baseline)", linewidth=2)
        ax1.semilogy(snr_range, results["ber_detnet"], "^-", label="Unfolded FSO-DetNet (Proposed)", linewidth=2.5)
        ax1.semilogy(snr_range, results["ber_ml"], "d:", label="Genie-Aided ML (Perfect CSI)", linewidth=2)
        ax1.set_xlabel("SNR (dB)")
        ax1.set_ylabel("Bit Error Rate (BER)")
        ax1.set_title(f"BER vs. SNR ({turbulence_regime} Turbulence, ρ={spatial_rho})")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.legend()
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        ax2.plot(snr_range, results["nmse_ls"], "o--", label="Least Squares (LS)", linewidth=2)
        ax2.plot(snr_range, results["nmse_dnn"], "s-.", label="Joint DNN (Baseline)", linewidth=2)
        ax2.plot(snr_range, results["nmse_detnet"], "^-", label="Unfolded FSO-DetNet (Proposed)", linewidth=2.5)
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("Channel NMSE (dB)")
        ax2.set_title("Channel Estimation NMSE vs. SNR")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

with tab2:
    avg_latencies = [
        np.mean(results["lat_lszf"]),
        np.mean(results["lat_dnn"]),
        np.mean(results["lat_detnet"]),
        np.mean(results["lat_ml"]),
    ]
    labels = ["LS-ZF", "Joint DNN", "Unfolded FSO-DetNet\n(Proposed)", "Genie-Aided ML\n(256 Exhaustive Search)"]

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    bars = ax3.bar(labels, avg_latencies, color=["#64748b", "#3b82f6", "#10b981", "#ef4444"])
    ax3.set_ylabel("Average Inference Latency (ms / frame)")
    ax3.set_title("Computational Efficiency: Per-Frame Detection Latency")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, avg_latencies):
        ax3.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f} ms", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig3)

with tab3:
    st.markdown(f"**Sample 8×8 Optical Channel Gain Matrix $H$ at SNR = {selected_snr} dB (Non-Negative $H \\geq 0$)**")
    hc1, hc2, hc3 = st.columns(3)
    vmax = float(np.max(snapshot["H_true"]))

    for col, title, mat in [
        (hc1, "True Channel H (Ground Truth)", snapshot["H_true"]),
        (hc2, "Initial LS Estimate", snapshot["H_ls"]),
        (hc3, "Unfolded FSO-DetNet Refined H", snapshot["H_det"]),
    ]:
        with col:
            fig_h, ax_h = plt.subplots(figsize=(4.2, 3.8))
            im = ax_h.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
            ax_h.set_title(title, fontsize=10)
            ax_h.set_xlabel("Optical Tx (1–8)")
            ax_h.set_ylabel("Optical Rx (1–8)")
            plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            st.pyplot(fig_h)


