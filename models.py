import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelRefinementModule(nn.Module):
    def __init__(self, num_tx=8, num_rx=8, hidden_dim=128):
        super().__init__()
        dim = num_rx * num_tx
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, H_ls):
        B, Nr, Nt = H_ls.shape
        flat = H_ls.reshape(B, -1)
        delta = self.net(flat).reshape(B, Nr, Nt)
        return F.relu(H_ls + delta)


class UnfoldedPGDLayer(nn.Module):
    def __init__(self, num_tx=8, num_rx=8, hidden_dim=64):
        super().__init__()
        self.step_size = nn.Parameter(torch.tensor(0.15))
        self.correction = nn.Sequential(
            nn.Linear(3 * num_tx, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tx)
        )

    def forward(self, x_k, H_est, y_data):
        Hx = torch.bmm(H_est, x_k)
        residual = Hx - y_data
        grad = torch.bmm(H_est.transpose(1, 2), residual)
        hty = torch.bmm(H_est.transpose(1, 2), y_data)
        feat = torch.cat([x_k.squeeze(-1), grad.squeeze(-1), hty.squeeze(-1)], dim=-1)
        neural_corr = self.correction(feat).unsqueeze(-1)
        z_next = x_k - F.softplus(self.step_size) * grad + 0.1 * neural_corr
        return torch.clamp(z_next, 0.0, 1.0)


class UnfoldedFSODetNet(nn.Module):
    def __init__(self, num_tx=8, num_rx=8, num_layers=6, responsivity=0.8):
        super().__init__()
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.responsivity = responsivity
        self.channel_refiner = ChannelRefinementModule(num_tx, num_rx, hidden_dim=128)
        self.pgd_layers = nn.ModuleList([
            UnfoldedPGDLayer(num_tx, num_rx, hidden_dim=64) for _ in range(num_layers)
        ])

    def estimate_ls_channel(self, Y_pilot, P):
        reg = 1e-3 * torch.eye(self.num_tx, device=P.device, dtype=P.dtype)
        P_pinv = torch.matmul(P.T, torch.linalg.inv(torch.matmul(P, P.T) + reg))
        H_ls = torch.matmul(Y_pilot, P_pinv.unsqueeze(0)) / self.responsivity
        return F.relu(H_ls)

    def forward(self, Y_pilot, Y_data, P):
        B = Y_pilot.shape[0]
        H_ls = self.estimate_ls_channel(Y_pilot, P)
        H_est = self.channel_refiner(H_ls)
        H_eff = self.responsivity * H_est
        x_k = torch.full((B, self.num_tx, 1), 0.5, device=Y_data.device, dtype=Y_data.dtype)
        for layer in self.pgd_layers:
            x_k = layer(x_k, H_eff, Y_data)
        return H_est, x_k


class JointDNN(nn.Module):
    def __init__(self, num_tx=8, num_rx=8, hidden_dim=256):
        super().__init__()
        self.num_tx = num_tx
        self.num_rx = num_rx
        in_dim = (num_rx * num_tx) + num_rx
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_h = nn.Linear(hidden_dim, num_rx * num_tx)
        self.head_x = nn.Linear(hidden_dim, num_tx)

    def forward(self, Y_pilot, Y_data, P=None):
        B = Y_pilot.shape[0]
        inp = torch.cat([Y_pilot.reshape(B, -1), Y_data.reshape(B, -1)], dim=-1)
        feat = self.backbone(inp)
        H_est = F.relu(self.head_h(feat)).reshape(B, self.num_rx, self.num_tx)
        x_est = torch.sigmoid(self.head_x(feat)).reshape(B, self.num_tx, 1)
        return H_est, x_est


def ls_zf_detector(Y_pilot, Y_data, P, responsivity=0.8, reg=1e-2):
    num_tx = P.shape[0]
    P_pinv = P.T @ np.linalg.inv(P @ P.T + 1e-3 * np.eye(num_tx))
    H_ls = np.maximum((Y_pilot @ P_pinv[None, :, :]) / responsivity, 0.0)
    B = Y_data.shape[0]
    X_hat = np.zeros((B, num_tx, 1), dtype=np.float32)
    eye = np.eye(num_tx, dtype=np.float32)
    for i in range(B):
        H_eff = responsivity * H_ls[i]
        W_zf = np.linalg.inv(H_eff.T @ H_eff + reg * eye) @ H_eff.T
        x_soft = W_zf @ Y_data[i]
        X_hat[i] = (x_soft >= 0.5).astype(np.float32)
    return H_ls, X_hat


def genie_ml_detector(H_true, Y_data, responsivity=0.8):
    num_tx = H_true.shape[2]
    candidates = np.array(list(itertools.product([0.0, 1.0], repeat=num_tx)), dtype=np.float32)
    B = Y_data.shape[0]
    X_ml = np.zeros((B, num_tx, 1), dtype=np.float32)
    for i in range(B):
        H_eff = responsivity * H_true[i]
        Y_cand = H_eff @ candidates.T
        dists = np.sum((Y_data[i] - Y_cand) ** 2, axis=0)
        best_idx = int(np.argmin(dists))
        X_ml[i, :, 0] = candidates[best_idx]
    return X_ml


def train_quick_models(transmit_fn, steps=220, batch_size=128):
    torch.manual_seed(42)
    detnet = UnfoldedFSODetNet(num_tx=8, num_rx=8, num_layers=6)
    dnn = JointDNN(num_tx=8, num_rx=8)
    opt_detnet = torch.optim.Adam(detnet.parameters(), lr=2e-3)
    opt_dnn = torch.optim.Adam(dnn.parameters(), lr=2e-3)
    detnet.train()
    dnn.train()
    for step in range(steps):
        snr = float(np.random.uniform(5.0, 25.0))
        H_true, P, Y_p, Y_d, X_true = transmit_fn(
            num_frames=batch_size, snr_db=snr, regime="Moderate", rho=0.5
        )
        H_t, P_t, Yp_t, Yd_t, X_t = (
            torch.tensor(H_true),
            torch.tensor(P),
            torch.tensor(Y_p),
            torch.tensor(Y_d),
            torch.tensor(X_true),
        )
        opt_detnet.zero_grad()
        H_pred, X_pred = detnet(Yp_t, Yd_t, P_t)
        loss_h = F.mse_loss(H_pred, H_t) / (torch.mean(H_t ** 2) + 1e-8)
        loss_det = 0.5 * loss_h + F.mse_loss(X_pred, X_t)
        loss_det.backward()
        opt_detnet.step()

        opt_dnn.zero_grad()
        H_pred_d, X_pred_d = dnn(Yp_t, Yd_t)
        loss_hd = F.mse_loss(H_pred_d, H_t) / (torch.mean(H_t ** 2) + 1e-8)
        loss_dnn = 0.5 * loss_hd + F.mse_loss(X_pred_d, X_t)
        loss_dnn.backward()
        opt_dnn.step()

    detnet.eval()
    dnn.eval()
    return detnet, dnn
