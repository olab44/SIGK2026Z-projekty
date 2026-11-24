"""Losses. LPIPS and FLIP are deliberately NOT used as losses (they are evaluation metrics)."""
import torch
import torch.nn.functional as F


def d_loss_lsgan(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())


def g_loss_lsgan(d_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((d_fake - 1) ** 2).mean()


def dilate_mask(mask: torch.Tensor, pixels: int = 2) -> torch.Tensor:
    """Dilate a (B,1,H,W) mask by `pixels` pixels (max-pooling)."""
    k = 2 * pixels + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pixels)


def weighted_l1(fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor, alpha: float = 20.0,
                dilation: int = 2) -> torch.Tensor:
    """L1 with weights w = 1 + alpha * dilate(mask), normalised by the sum of weights
    (so the loss scale does not depend on the sphere size)."""
    w = 1.0 + alpha * dilate_mask(mask, dilation)
    err = (fake - real).abs() * w
    return err.sum() / (w.sum() * fake.shape[1])


def generator_loss(fake_img, mask_logit, real, mask, d_fake=None, lambda_adv=1.0, lambda_l1=100.0,
                   lambda_mask=2.0, alpha=20.0):
    """lambda_adv * LSGAN + lambda_l1 * mask-weighted L1 + lambda_mask * BCE(mask).

    With d_fake=None or lambda_adv=0 the adversarial term is dropped (L1-only baseline).
    """
    l1 = weighted_l1(fake_img, real, mask, alpha)
    bce = F.binary_cross_entropy_with_logits(mask_logit, mask)
    adv = g_loss_lsgan(d_fake) if (d_fake is not None and lambda_adv > 0) else fake_img.new_zeros(())
    total = lambda_adv * adv + lambda_l1 * l1 + lambda_mask * bce
    return total, {"g_adv": adv.item(), "g_l1": l1.item(), "g_mask": bce.item()}
