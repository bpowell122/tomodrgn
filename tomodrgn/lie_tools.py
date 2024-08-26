"""
Tools for dealing with SO(3) group and algebra
Adapted from https://github.com/pimdh/lie-vae
All functions are pytorch-ified
"""

import numpy as np
import torch
from torch.distributions import Normal


def map_to_lie_algebra(v: torch.Tensor) -> torch.Tensor:
    """
    Map a point in R^N to the tangent space at the identity, i.e. to the Lie Algebra
    :param v: vector in R^N, (..., 3) in our case
    :return: v converted to Lie Algebra element, (3,3) in our case
    """
    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    rot_x = v.new_tensor([[0., 0., 0.],
                          [0., 0., -1.],
                          [0., 1., 0.]])

    rot_y = v.new_tensor([[0., 0., 1.],
                          [0., 0., 0.],
                          [-1., 0., 0.]])

    rot_z = v.new_tensor([[0., -1., 0.],
                          [1., 0., 0.],
                          [0., 0., 0.]])

    rot = rot_x * v[..., 0, None, None] + rot_y * v[..., 1, None, None] + rot_z * v[..., 2, None, None]
    return rot


def expmap(v: torch.Tensor) -> torch.Tensor:
    """

    :param v:
    :return:
    """
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    rot = map_to_lie_algebra(v / theta)

    identity = torch.eye(3, device=v.device, dtype=v.dtype)
    rot = identity + torch.sin(theta)[..., None] * rot + (1. - torch.cos(theta))[..., None] * (rot @ rot)
    return rot


def s2s1rodrigues(s2_el: torch.Tensor,
                  s1_el: torch.Tensor) -> torch.Tensor:
    """

    :param s2_el:
    :param s1_el:
    :return:
    """
    rot = map_to_lie_algebra(s2_el)
    cos_theta = s1_el[..., 0]
    sin_theta = s1_el[..., 1]
    identity = torch.eye(3, device=s2_el.device, dtype=s2_el.dtype)
    rot = identity + sin_theta[..., None, None] * rot + (1. - cos_theta)[..., None, None] * (rot @ rot)
    return rot


def s2s2_to_SO3(v1: torch.Tensor,
                v2: torch.Tensor | None = None) -> torch.Tensor:
    """
    Normalize 2 3-vectors.
    Project second to orthogonal component.
    Take cross product for third.
    Stack to form SO matrix.
    :param v1:
    :param v2:
    :return:
    """
    if v2 is None:
        assert v1.shape[-1] == 6
        v2 = v1[..., 3:]
        v1 = v1[..., 0:3]
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)


def SO3_to_s2s2(r: torch.Tensor) -> torch.Tensor:
    """
    Map batch of SO(3) matrices to s2s2 representation as first two basis vectors, concatenated as Bx6
    :param r: SO3 rotation matrices
    :return:
    """
    return r.view(*r.shape[:-2], 9)[..., :6].contiguous()


def SO3_to_quaternions(r: torch.Tensor) -> torch.Tensor:
    """
    Map batch of SO(3) matrices to quaternions.
    :param r: SO3 rotation matrices
    :return: equivalent quaternions
    """
    batch_dims = r.shape[:-2]
    assert list(r.shape[-2:]) == [3, 3], 'Input must be 3x3 matrices'
    r = r.view(-1, 3, 3)
    n = r.shape[0]

    diags = [r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]]
    denom_pre = torch.stack([
        1 + diags[0] - diags[1] - diags[2],
        1 - diags[0] + diags[1] - diags[2],
        1 - diags[0] - diags[1] + diags[2],
        1 + diags[0] + diags[1] + diags[2]
    ], 1)
    denom = 0.5 * torch.sqrt(1E-6 + torch.abs(denom_pre))

    case0 = torch.stack([
        denom[:, 0],
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 0]),
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 0]),
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 0])
    ], 1)
    case1 = torch.stack([
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 1]),
        denom[:, 1],
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 1]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 1])
    ], 1)
    case2 = torch.stack([
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 2]),
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 2]),
        denom[:, 2],
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 2])
    ], 1)
    case3 = torch.stack([
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 3]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 3]),
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 3]),
        denom[:, 3]
    ], 1)

    cases = torch.stack([case0, case1, case2, case3], 1)

    quaternions = cases[torch.arange(n, dtype=torch.long), torch.argmax(denom.detach(), 1)]
    return quaternions.view(*batch_dims, 4)


def quaternions_to_SO3(q: torch.Tensor) -> torch.Tensor:
    """
    Normalizes q and maps to group matrix.
    :param q: input quaternion
    :return: equivalent SO3 rotation matrix
    """
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        r * r - i * i - j * j + k * k, 2 * (r * i + j * k), 2 * (r * j - i * k),
        2 * (r * i - j * k), -r * r + i * i - j * j + k * k, 2 * (i * j + r * k),
        2 * (r * j + i * k), 2 * (i * j - r * k), -r * r - i * i + j * j + k * k
    ], -1).view(*q.shape[:-1], 3, 3)


def random_quaternions(n: int,
                       dtype: torch.dtype = torch.float32,
                       device: torch.device | None = None) -> torch.Tensor:
    """
    Generate random rotations as quaternions
    :param n: number of rotations to generate
    :param dtype: dtype of output tensor
    :param device: device for output tensor
    :return: random quaternions tensor
    """
    u1, u2, u3 = torch.rand(3, n, dtype=dtype, device=device)
    return torch.stack((
        torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2),
        torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * np.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * np.pi * u3),
    ), 1)


def random_SO3(n: int,
               dtype: torch.dtype = torch.float32,
               device: torch.device | None = None) -> torch.Tensor:
    """
    Generate random rotations as SO3 matrices
    :param n: number of rotations to generate
    :param dtype: dtype of output tensor
    :param device: device for output tensor
    :return: random SO3 rotations tensor
    """
    return quaternions_to_SO3(random_quaternions(n, dtype, device))


def logsumexp(inputs: torch.Tensor,
              dim: int | None = None,
              keepdim: bool = False) -> torch.Tensor:
    """
    Numerically stable logsumexp.
    https://github.com/pytorch/pytorch/issues/2591
    :param inputs: A Variable with any shape.
    :param dim: An integer.
    :param keepdim: A boolean.
    :return: Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def so3_entropy_old(w_eps: torch.Tensor,
                    std: torch.Tensor,
                    k: int = 10) -> torch.Tensor:
    """

    :param w_eps: (Tensor of dim 3): sample from so3
    :param std: (Tensor of dim 3x3): covariance of distribution on so3
    :param k: 2k+1 samples for truncated summation
    :return:
    """
    # entropy of gaussian distribution on so3
    # see appendix C of https://arxiv.org/pdf/1807.04689.pdf
    theta = w_eps.norm(p=2)
    u = w_eps / theta  # 3
    angles = 2 * np.pi * torch.arange(-k, k + 1, dtype=w_eps.dtype, device=w_eps.device)  # 2k+1
    theta_hat = theta + angles  # 2k+1
    x = u[None, :] * theta_hat[:, None]  # 2k+1 , 3
    log_p = Normal(torch.zeros(3, device=w_eps.device), std).log_prob(x)  # 2k+1,3
    clamp = 1e-3
    log_vol = torch.log((theta_hat ** 2).clamp(min=clamp) / (2 - 2 * torch.cos(theta)).clamp(min=clamp))  # 2k+1
    log_p = log_p.sum(-1) + log_vol
    entropy = -logsumexp(log_p)
    return entropy


def so3_entropy(w_eps: torch.Tensor,
                std: torch.Tensor,
                k: int = 10) -> torch.Tensor:
    """

    :param w_eps: (Tensor of dim Bx3): sample from so3
    :param std: (Tensor of dim Bx3): std of distribution on so3
    :param k: Use 2k+1 samples for truncated summation
    :return:
    """
    # entropy of gaussian distribution on so3
    # see appendix C of https://arxiv.org/pdf/1807.04689.pdf
    theta = w_eps.norm(p=2, dim=-1, keepdim=True)  # [B, 1]
    u = w_eps / theta  # [B, 3]
    angles = 2 * np.pi * torch.arange(-k, k + 1, dtype=w_eps.dtype, device=w_eps.device)  # 2k+1
    theta_hat = theta[:, None, :] + angles[:, None]  # [B, 2k+1, 1]
    x = u[:, None, :] * theta_hat  # [B, 2k+1 , 3]
    log_p = Normal(torch.zeros(3, device=w_eps.device), std).log_prob(x.permute([1, 0, 2]))  # [2k+1, B, 3]
    log_p = log_p.permute([1, 0, 2])  # [B, 2k+1, 3]
    clamp = 1e-3
    log_vol = torch.log((theta_hat ** 2).clamp(min=clamp) / (2 - 2 * torch.cos(theta_hat)).clamp(min=clamp))  # [B, 2k+1, 1]
    log_p = log_p.sum(-1) + log_vol.sum(-1)  # [B, 2k+1]
    entropy = -logsumexp(log_p, -1)
    return entropy
