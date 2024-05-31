"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/rigid_utils.py
"""

import torch


def skew(w: torch.Tensor) -> torch.Tensor:
    """Build a skew matrix ("cross product matrix") for vector w.

    Modern Robotics Eqn 3.30.

    Args:
      w: (N, 3) A 3-vector

    Returns:
      W: (N, 3, 3) A skew matrix such that W @ v == w x v
    """
    zeros = torch.zeros(w.shape[0], device=w.device)
    w_skew_list = [zeros, -w[:, 2], w[:, 1],
                   w[:, 2], zeros, -w[:, 0],
                   -w[:, 1], w[:, 0], zeros]
    w_skew = torch.stack(w_skew_list, dim=-1).reshape(-1, 3, 3)
    return w_skew


def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.

    Args:
      R: (3, 3) An orthonormal rotation matrix.
      p: (3,) A 3-vector representing an offset.

    Returns:
      X: (4, 4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device).repeat(R.shape[0], 1, 1)
    transform = torch.cat([torch.cat([R, p], dim=-1), bottom_row], dim=1)

    return transform


def exp_so3(w: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
      w: (3,) An axis of rotation.
      theta: An angle of rotation.

    Returns:
      R: (3, 3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.
    """
    W = skew(w)
    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)  # batch matrix multiplication
    R = identity + torch.sin(theta.unsqueeze(-1)) * W + (1.0 - torch.cos(theta.unsqueeze(-1))) * W_sqr
    return R


def exp_se3(S: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.88.

    Args:
      S: (6,) A screw axis of motion.
      theta: Magnitude of motion.

    Returns:
      a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(w, theta)

    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)
    theta = theta.view(-1, 1, 1)

    p = torch.bmm((theta * identity + (1.0 - torch.cos(theta)) * W + (theta - torch.sin(theta)) * W_sqr),
                  v.unsqueeze(-1))
    return rp_to_se3(R, p)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    """Converts a vector to a homogeneous coordinate vector by appending a 1.

    Args:
        v: A tensor representing a vector or batch of vectors.

    Returns:
        A tensor with an additional dimension set to 1.
    """
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v: torch.Tensor) -> torch.Tensor:
    """Converts a homogeneous coordinate vector to a standard vector by dividing by the last element.

    Args:
        v: A tensor representing a homogeneous coordinate vector or batch of homogeneous coordinate vectors.

    Returns:
        A tensor with the last dimension removed.
    """
    return v[..., :3] / v[..., -1:]
