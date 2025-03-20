import torch.nn.functional as F
from torchvision.transforms import Resize
from piq import psnr, ssim, brisque, LPIPS

def validation_scores(model, HR, LR):
    SR = model.sample(LR)

    psnr_val = psnr(HR, SR, data_range=1.0).item()

    ssim_val = ssim(HR, SR, data_range=1.0, reduction='mean').item()

    lpips = LPIPS()
    lpips_val = lpips(HR, SR).item()

    brisque_val = brisque(SR).item()

    SR_downsampled = Resize((LR.size(2), LR.size(3)))(SR)
    SR_downsampled = SR_downsampled.clamp(0, 1)

    psnr_consistency_val = psnr(LR, SR_downsampled, data_range=1.0).item()

    mse_diversity_val = 0
    lpips_diversity_val = 0
    n_iterations = 32
    n_samples = 8
    for i in range(n_iterations):
        LR_input = LR[i].unsqueeze(0)
        LR_input = LR_input.repeat(n_samples, 1, 1, 1)
        HR_reference = HR[i].unsqueeze(0)
        HR_reference = HR_reference.repeat(n_samples, 1, 1, 1)
        SR = model.sample(LR_input)
        # torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')?
        for sample in SR:
            SR_ref = sample.unsqueeze(0)
            SR_ref = SR_ref.repeat(n_samples, 1, 1, 1)
            mse_diversity_val += F.mse_loss(SR_ref, SR, reduction='mean').item()
            lpips_diversity_val += lpips(SR_ref, SR).item()
    mse_diversity_val /= n_iterations*n_samples
    lpips_diversity_val /= n_iterations*n_samples

    return psnr_val, ssim_val, lpips_val, brisque_val, psnr_consistency_val, mse_diversity_val, lpips_diversity_val
