import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["barlow_loss", "byol_loss", "moco_loss", "simclr_loss", "simsiam_loss", "vicreg_loss"]


def simclr_loss(
    projections: torch.Tensor,
    projections_norm: bool = True,
    temperature: float = 0.15,
) -> torch.Tensor:
    """
    ``simclr_loss`` computes the normalized temperature-scaled cross entropy
    loss [NTXent]_ , which is used in many contrastive learning algorithm.
    It is basically a simple implementation of the InfoNCE_loss provided in the
    official simCLR repository [simgit]_ using only torch functions.

    Parameters
    ----------
    projections: torch.Tensor
        2D Tensor where projections[0:N/2] are the projections of one batch
        augmented version and projections[N/2:] are the projections of the other
        batch augmented version
    projections_norm: bool, optional
        Whether to normalize the projections or not.

        Default = True
    temperature: float, optional
        Temperature coefficient of the NTX_ent loss
        (See references to check loss formula).

        Default = 0.15

    Returns
    -------
    loss: torch.Tensor
        The calculated loss.

    Note
    ----
    Looking at some implementations (e.g. the one in lightlyAI),
    the returned loss seems to be double. However the function
    add_contrastive_loss in the original repo returns the same value as this
    implementation.

    References
    ----------
    .. [NTXent] Chen et al. A Simple Framework for Contrastive Learning of Visual
      Representations. (2020). https://doi.org/10.48550/arXiv.2002.05709
    .. [simgit] To check the original tensorflow implementation visit the
      following repository: https://github.com/google-research/simclr
      (look at the function add_contrastive_loss in objective.py)

    Example
    -------
    >>> import torch
    >>> import selfeeg.losses
    >>> torch.manual_seed(1234)
    >>> projections = torch.randn(64, 32)
    >>> loss = losses.simclr_loss(projections)
    >>> print(loss) # will return 10.2866

    """
    if projections_norm:
        # L2 norm along first dimension
        projections = F.normalize(projections, p=2.0, dim=1)

    proj1, proj2 = torch.split(projections, int(projections.shape[0] / 2))
    N = proj1.shape[0]
    labels = torch.eye(N, N * 2).to(device=projections.device)
    masks = torch.eye(N).to(device=projections.device)

    nn = torch.matmul(proj1, torch.transpose(proj1, 0, 1)) / temperature
    nn = nn - (masks * 1e9)
    mm = torch.matmul(proj2, torch.transpose(proj2, 0, 1)) / temperature
    mm = mm - (masks * 1e9)
    nm = torch.matmul(proj1, torch.transpose(proj2, 0, 1)) / temperature
    mn = torch.matmul(proj2, torch.transpose(proj1, 0, 1)) / temperature

    loss_1 = F.cross_entropy(torch.cat([nm, nn], 1), labels, reduction="mean")
    loss_2 = F.cross_entropy(torch.cat([mn, mm], 1), labels, reduction="mean")
    loss = loss_1 + loss_2

    return loss


def simsiam_loss(
    p1: torch.Tensor,
    z1: torch.Tensor,
    p2: torch.Tensor,
    z2: torch.Tensor,
    projections_norm: bool = True,
) -> torch.Tensor:
    """
    Simple implementation of the SimSiam [simsiam]_ loss function with
    the possibility to not normalize tensors. Official repo can be found here
    [siamgit]_

    Parameters
    ----------
    p1: torch.Tensor
        2D Tensor with one augmented batch predictor output.
    z1: torch.Tensor
        2D Tensor with one augmented batch projection output.
    p2: torch.Tensor
        Same as p1 but with the other augmented batch.
    z2: torch.Tensor
        Same as z1 with the other augmented batch.
    projections_norm: bool, optional
        Whether to normalize the projections or not.

        Default= True

    Returns
    -------
    loss: torch.Tensor
        The calculated loss.

    References
    ----------
    .. [siamgit] Original github repo: https://github.com/facebookresearch/simsiam
    .. [simsiam] Original paper: Chen & He. Exploring Simple Siamese Representation
      Learning. https://arxiv.org/abs/2011.10566

    Example
    -------
    >>> import torch
    >>> import selfeeg.losses
    >>> torch.manual_seed(1234)
    >>> p1 = torch.randn(64, 32)
    >>> z1 = torch.randn(64, 32)
    >>> p2 = torch.randn(64, 32)
    >>> z2 = torch.randn(64, 32)
    >>> loss = losses.simsiam_loss(p1,z1,p2,z2)
    >>> print(loss) # will return -0.0161

    """

    if projections_norm:
        p1 = F.normalize(p1, p=2.0, dim=1)
        z1 = F.normalize(z1.detach(), p=2.0, dim=1)
        p2 = F.normalize(p2, p=2.0, dim=1)
        z2 = F.normalize(z2.detach(), p=2.0, dim=1)

    D1 = -(p1 * z2).sum(dim=1).mean()
    D2 = -(p2 * z1).sum(dim=1).mean()
    loss = 0.5 * D1 + 0.5 * D2
    return loss


def moco_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    queue: torch.Tensor = None,
    projections_norm: bool = True,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Simple implementation of the MoCo loss function [moco2]_.
    It is the InfoNCE loss with dot product as similarity and memory bank as
    negative samples. If no queue related to the memory bank is given, MoCo v3
    [moco3]_ loss calculation is performed. Note that the real MoCo v3 loss is
    calculated by calling the function 2 times (with different q and k tensors)
    and summing up the results.

    Parameters
    ----------
    q: torch.Tensor
        2D (NxC) Tensor with the queries, i.e. one augmented batch predictor or
        projection_head output. N = batch size, C = number of features.
    k: torch.Tensor
        2D (NxC) Tensor with the keys, i.e. one augmented batch projection_head
        output which will be added to the memory bank.
        N = batch size , C = number of features.
    queue:  torch.Tensor, optional
        2D (CxK) Tensor with the memory bank, i.e. a collection of previous
        augmented batch projection_head outputs which act as negative samples.
        C = number of features, K = memory bank size.

        Default = None
    projections_norm: bool, optional
        Whether to normalize the projections or not.

        Default = True
    temperature: float, optional
        Temperature coefficient of the NTX_ent loss.

        Default = 0.15

    Returns
    -------
    loss: torch.Tensor
        The calculated loss.

    References
    ----------
    .. [moco2] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick,
      “Momentum contrast for unsupervised visual representation learning,”
      in Proceedings of the IEEE/CVF conference on computer vision and pattern
      recognition, pp. 9729–9738, 2020.
    .. [moco3] X. Chen, H. Fan, R. Girshick, and K. He, “Improved baselines with
      momentum contrastive learning,” arXiv preprint arXiv:2003.04297, 2020.

    Example
    -------
    >>> import torch
    >>> import selfeeg.losses
    >>> torch.manual_seed(1234)
    >>> q = torch.randn(64, 32)
    >>> k = torch.randn(64, 32)
    >>> queue = torch.randn(32, 128)
    >>> loss = losses.moco_loss(q, k, queue)
    >>> print(loss) # will return 17.1668
    >>> loss = losses.moco_loss(q, k)
    >>> print(loss) # will return 1.4349

    """

    N, C = q.shape
    # normalize
    if projections_norm:
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

    # if no queue is given, run MoCo v3 loss
    # (note that MoCo v3 is MoCo_loss(q1,k2) + MoCo_loss(q2,k1)
    if queue == None:
        logits = torch.einsum("nc,mc->nm", [q, k]) / temperature
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long, device=logits.device)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * temperature)

    # positive logits: Nx1
    l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(-1)
    # negative logits: NxK
    l_neg = torch.matmul(q, queue.detach())
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)
    # apply temperature
    logits /= temperature
    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    return loss


def byol_loss(
    p1: torch.Tensor,
    z1: torch.Tensor,
    p2: torch.Tensor,
    z2: torch.Tensor,
    projections_norm: bool = True,
) -> torch.Tensor:
    """
    Simple pytorch implementation of the BYOL loss function presented in [BYOL]_ .

    Parameters
    ----------
    p1: torch.Tensor
        2D Tensor with one augmented batch predictor output.
    z1: torch.Tensor
        2D Tensor with one augmented batch projection output.
    p2: torch.Tensor
        Same as p1 but with the other augmented batch.
    z2: torch.Tensor
        Same as z1 with the other augmented batch.
    projections_norm: bool, optional
        Whether to normalize the projections or not.

        Default= True

    Returns
    -------
    loss: torch.Tensor
        The calculated loss.

    References
    ----------
    .. [BYOL] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond,
      E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar,
      et al., “Bootstrap your own latent - a new approach to self-supervised
      learning,” Advances in neural information processing systems,
      vol. 33, pp. 21271– 21284, 2020.

    Example
    -------
    >>> import torch
    >>> import selfeeg.losses
    >>> torch.manual_seed(1234)
    >>> p1 = torch.randn(64, 32)
    >>> z1 = torch.randn(64, 32)
    >>> p2 = torch.randn(64, 32)
    >>> z2 = torch.randn(64, 32)
    >>> loss = losses.byol_loss(p1,z1,p2,z2)
    >>> print(loss) # will return 3.9357

    """

    if projections_norm:
        p1 = F.normalize(p1, p=2.0, dim=1)
        z1 = F.normalize(z1.detach(), p=2.0, dim=1)
        p2 = F.normalize(p2, p=2.0, dim=1)
        z2 = F.normalize(z2.detach(), p=2.0, dim=1)

    loss1 = 2 - 2 * (p1 * z2).sum(dim=-1)
    loss2 = 2 - 2 * (p2 * z1).sum(dim=-1)
    loss = loss1 + loss2
    return loss.mean()


def barlow_loss(
    z1: torch.Tensor,
    z2: torch.Tensor = None,
    lambda_coeff: float = 5e-3,
) -> torch.Tensor:
    """
    Pytorch implementation of the Barlow Twins loss function
    as presented in [barlow]_ .

    Parameters
    ----------
    z1: torch.tensor
        2D tensor with projections of one augmented version of the batch.
    z2: torch.tensor, optional
        2D projections of the other augmented version of the batch. Can be none if
        z1 and z2 are concatenated. In this case internal split is done.

        Default = None
    lambda_coeff: float, optional
        Off diagonal scaling factor described in the paper.

        Default = 5e-3

    Returns
    -------
    loss: torch.Tensor
        The calculated loss.

    References
    ----------
    .. [barlow] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny,
      “Barlow twins: Self-supervised learning via redundancy reduction,”
      in International Conference on Machine Learning, pp. 12310–12320, PMLR, 2021.

    Example
    -------
    >>> import torch
    >>> import selfeeg.losses
    >>> torch.manual_seed(1234)
    >>> z1 = torch.randn(64, 32)
    >>> z2 = torch.randn(64, 32)
    >>> loss = losses.barlow_loss(z1,z2)
    >>> print(loss) # will return 31.6141

    """
    if z2 == None:
        z1, z2 = torch.split(z1, int(z1.shape[0] / 2))

    N, D = z1.shape
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)

    c_mat = (z1_norm.T @ z2_norm) / N
    c_mat2 = c_mat.pow(2)

    loss = (
        D
        - 2 * torch.trace(c_mat)
        + lambda_coeff * torch.sum(c_mat**2)
        + (1 - lambda_coeff) * torch.trace(c_mat**2)
    )
    return loss


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor = None,
    Lambda: float = 25,
    Mu: float = 25,
    Nu: float = 1,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    Pytorch implementation of the VICReg loss function [VIC]_ .

    Parameters
    ----------
    z1: torch.tensor
        2D tensor with projections of one augmented version of the batch.
    z2: torch.tensor, optional
        2D projections of the other augmented version of the batch. Can be none if
        z1 and z2 are cat together. In this case internal split is done,
        but be sure that the first dimension can be divided by 2.

        Default = None
    Lambda: float, optional
        Coefficient applied to the invariant loss.

        Default = 25
    Mu: float, optional
        Coefficient applied to the variance loss .

        Default = 25
    Nu: float, optional
        Coefficient applied to the covariance.

        Default = 1
    epsilon: float, optional
        Value summed to the variance for stability purposes.

        Default = 1e-4

    Returns
    -------
    loss: torch.Tensor
        The calculated loss.

    References
    ----------
    .. [VIC] A. Bardes, J. Ponce, and Y. LeCun,
      “Vicreg: Variance-invariance-covariance regularization for self-supervised
      learning,” arXiv preprint arXiv:2105.04906, 2021.

    Example
    -------
    >>> import torch
    >>> import selfeeg.losses
    >>> torch.manual_seed(1234)
    >>> z1 = torch.randn(64, 32)
    >>> z2 = torch.randn(64, 32)
    >>> loss = losses.vicreg_loss(z1,z2)
    >>> print(loss) # will return 53.0773

    """
    if z2 == None:
        z1, z2 = torch.split(z1, int(z1.shape[0] / 2))

    N, D = z1.shape
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    # invariance loss
    sim_loss = F.mse_loss(z1, z2)

    # variance loss
    std_z1 = torch.sqrt(z1.var(dim=0) + epsilon)
    std_z2 = torch.sqrt(z2.var(dim=0) + epsilon)
    std_loss = (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))) / 2

    # covariance loss
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z1[range(D), range(D)] = 0.0
    cov_z2 = (z2.T @ z2) / (N - 1)
    cov_z2[range(D), range(D)] = 0.0
    cov_loss = cov_z1.pow_(2).sum() / D + cov_z2.pow_(2).sum() / D
    loss = Lambda * sim_loss + Mu * std_loss + Nu * cov_loss
    return loss
