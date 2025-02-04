# gameapi/model.py
import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer

class QuantumMultiHeadAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, x, n_heads, head_dim, epsilon=1e-3):
        theta_np = theta.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        ctx.save_for_backward(theta, x)
        ctx.epsilon = epsilon
        ctx.n_heads = n_heads
        ctx.head_dim = head_dim

        outs = []
        for sample in x_np:  # shape: (n_heads, head_dim)
            out_vals = _run_multihead_qiskit_circuit(theta_np, sample)
            outs.append(out_vals)  # shape: (n_heads,)
        outs = np.stack(outs)  # shape: (batch_size, n_heads)
        return torch.tensor(outs, device=theta.device, dtype=torch.float32).unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        theta, x = ctx.saved_tensors
        epsilon = ctx.epsilon
        n_heads = ctx.n_heads
        head_dim = ctx.head_dim

        theta_np = theta.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        grad_theta = torch.zeros_like(theta)
        grad_output_np = grad_output.detach().cpu().numpy()  # shape: (batch_size, n_heads, 1)

        for i in range(len(theta_np)):
            theta_up = np.copy(theta_np)
            theta_up[i] += epsilon
            theta_down = np.copy(theta_np)
            theta_down[i] -= epsilon

            outs_up, outs_down = [], []
            for sample in x_np:
                outs_up.append(_run_multihead_qiskit_circuit(theta_up, sample))
                outs_down.append(_run_multihead_qiskit_circuit(theta_down, sample))
            outs_up = np.stack(outs_up)
            outs_down = np.stack(outs_down)
            grad_i = (outs_up - outs_down) / (2 * epsilon)
            grad_i = grad_i[:, :, None] * grad_output_np
            grad_theta[i] = np.sum(grad_i)
        return grad_theta, None, None, None, None

def _run_multihead_qiskit_circuit(theta_vals, sample):
    n_heads, head_dim = sample.shape
    n_qubits = n_heads * head_dim
    qc = QuantumCircuit(n_qubits)
    flat_data = sample.reshape(-1)
    for i, val in enumerate(flat_data):
        qc.ry(val, i)
    required = n_qubits * 3
    if len(theta_vals) < required:
        raise ValueError("Insufficient theta parameters for circuit design")
    idx = 0
    for q in range(n_qubits):
        qc.rz(theta_vals[idx], q)
        qc.ry(theta_vals[idx+1], q)
        qc.rz(theta_vals[idx+2], q)
        idx += 3
    for q in range(n_qubits - 1):
        qc.cx(q, q+1)
    backend = Aer.get_backend("statevector_simulator")
    qc.name = "unique_name"
    job = backend.run(qc, shots=1)
    result = job.result()
    sv = result.get_statevector(qc)
    out_vals = []
    for h in range(n_heads):
        qubit_idx = h * head_dim
        out_vals.append(_z_exp_qubit(sv, qubit_idx, n_qubits))
    return np.array(out_vals)

def _z_exp_qubit(statevector, qubit_idx, total_qubits):
    probs = np.abs(statevector)**2
    exp_z = 0.0
    for basis_idx, p in enumerate(probs):
        bit_val = (basis_idx >> qubit_idx) & 1
        exp_z += p if bit_val == 0 else -p
    return exp_z

class QuantumDifficultyAdjuster(nn.Module):
    def __init__(self, n_heads=2, head_dim=2, num_classes=3):
        super(QuantumDifficultyAdjuster, self).__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.num_classes = num_classes
        n_qubits = n_heads * head_dim  # Should be 4 for our 4-dim input (e.g., n_heads=2, head_dim=2)
        self.n_params = n_qubits * 3
        self.theta = nn.Parameter(0.1 * torch.randn(self.n_params))
        # A simple linear layer mapping the quantum measurement (per head) to class logits.
        self.fc = nn.Linear(n_heads, num_classes)

    def forward(self, x):
        # x: (batch_size, 4)
        bsz = x.shape[0]
        x_reshaped = x.view(bsz, self.n_heads, self.head_dim)
        q_out = QuantumMultiHeadAttentionFunction.apply(self.theta, x_reshaped, self.n_heads, self.head_dim)
        q_out = q_out.squeeze(-1)  # (bsz, n_heads)
        logits = self.fc(q_out)    # (bsz, num_classes)
        return logits
