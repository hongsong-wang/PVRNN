import torch
import torch.nn.functional as F
import numpy as np

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.size(-1) == 3
    theta = torch.norm(e, p=2, dim=-1, keepdim=True)
    w = torch.cos(0.5 * theta)
    xyz = torch.sin(0.5 * theta)/(theta + 1e-6) * e
    q = torch.cat([w, xyz], dim=-1)
    return q

def quaternion_to_expmap(q):
    """
      Converts an exponential map angle to a rotation matrix
      Matlab port to python for evaluation purposes
      I believe this is also called Rodrigues' formula
      https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    q is (*, 4)
    return (*, 3)
    examples:
        e = torch.rand(1, 3, 3)
        q = expmap_to_quaternion(e)
        e2 = quaternion_to_expmap(q)
    """
    sinhalftheta = torch.index_select(q, dim=-1, index=torch.tensor([1,2,3]).to(q.device))
    coshalftheta = torch.index_select(q, dim=-1, index=torch.tensor([0]).to(q.device))

    norm_sin = torch.norm(sinhalftheta, p=2, dim=-1, keepdim=True)
    r0 = torch.div(sinhalftheta, norm_sin)

    theta = 2 * torch.atan2(norm_sin, coshalftheta)
    theta = torch.fmod(theta + 2 * np.pi, 2 * np.pi)

    theta = torch.where(theta > np.pi, 2 * np.pi - theta, theta)
    r0 = torch.where(theta > np.pi, -r0, r0)
    r = r0 * theta
    return r

def qeuler(q, order='zyx', epsilon=1e-6):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    q0 = torch.index_select(q, dim=-1, index=torch.tensor([0]).to(q.device) )
    q1 = torch.index_select(q, dim=-1, index=torch.tensor([1]).to(q.device))
    q2 = torch.index_select(q, dim=-1, index=torch.tensor([2]).to(q.device))
    q3 = torch.index_select(q, dim=-1, index=torch.tensor([3]).to(q.device))

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise('not defined')

    return torch.cat([x, y, z], dim=-1)

def tensor_expmap_to_euler(data):
    # data is (*, feature_dim), feature_dim is multiple of 3
    ori_shp = data.size()
    eul = qeuler(expmap_to_quaternion(data.contiguous().view(-1, 3)) )
    return eul.view(ori_shp)

def tensor_expmap_to_quaternion(data, channel_first=False):
    ori_shp = list(data.shape)
    new_shp = ori_shp[0:-1] + [int(ori_shp[-1] / 3 * 4)]
    qu = expmap_to_quaternion(data.contiguous().view(-1, 3))

    if channel_first:
        ori_shp = ori_shp[0:-1] + [int(ori_shp[-1] / 3), 4]
        qu = qu.view(ori_shp).transpose(-1, -2).contiguous()
    return qu.view(new_shp)

def get_train_expmap_to_quaternion_loss(output, target):
    # data is (time, batch, dim)
    out = tensor_expmap_to_quaternion(output)
    trg = tensor_expmap_to_quaternion(target)
    if 1:
        distance = out - trg
    else:
        out = out.view(out.size(0), out.size(1), out.size(2)/4, 4)
        trg = trg.view(trg.size(0), trg.size(1), trg.size(2)/4, 4)
        distance = 1 - torch.pow(torch.sum(out*trg, dim=-1), 2)

    # Todo, it seems that abs better than pow2/sqrt for quaternion
    return torch.mean(torch.abs(distance))
    # return torch.mean(torch.sqrt(torch.sum(torch.pow(distance, 2), -1)) )

if __name__ == '__main__':
    pass