import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM adapted for a regression/generation target instead of a classification
    logit -- backprops from the image reconstruction loss instead of a class score,
    on the visual encoder's last conv layer (content_backbone.encoder_conv).

    Standard Grad-CAM assumes a classifier with a class score to backprop from. Here
    there's no class score, so the "target" is the L1 loss between the predicted and
    target image. The gradient tells us which regions of the encoder's last feature
    map, if changed, would most affect that loss -- which is the same idea Grad-CAM
    uses, just applied to the reconstruction loss instead of a logit.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute(self, image_seq, text_seq, target_seq, image_target, criterion, frame_idx, batch_idx=0):
        """Runs a forward+backward pass and returns a (H, W) heatmap in [0, 1] for the
        given input frame_idx (0..S-1) of the given batch_idx.
        """
        self.model.zero_grad()
        pred_img_content, _, _, _, _, _, _ = self.model(image_seq, text_seq, target_seq)
        loss = criterion(pred_img_content, image_target)
        loss.backward()

        B, S = image_seq.shape[:2]
        flat_idx = batch_idx * S + frame_idx

        acts = self.activations[flat_idx]        # (C, h, w)
        grads = self.gradients[flat_idx]          # (C, h, w)
        weights = grads.mean(dim=(1, 2))          # (C,)

        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (h, w)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        target_hw = image_seq.shape[-2:]  # (H, W) of the actual input frame
        cam = F.interpolate(cam[None, None], size=target_hw, mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu(), loss.item()


def overlay_heatmap(frame_chw, heatmap, alpha=0.5, cmap="jet"):
    """frame_chw: (3, H, W) tensor in [0,1]. heatmap: (H, W) tensor in [0,1].
    Returns an (H, W, 3) numpy array ready for plt.imshow."""
    import matplotlib.cm as cm
    import numpy as np

    frame_np = frame_chw.permute(1, 2, 0).detach().cpu().numpy()
    heatmap_np = heatmap.numpy()
    colored = cm.get_cmap(cmap)(heatmap_np)[:, :, :3]
    return np.clip((1 - alpha) * frame_np + alpha * colored, 0, 1)
