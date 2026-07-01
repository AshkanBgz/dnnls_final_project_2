import torch
import torch.nn as nn
import torch.nn.functional as F


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion_images,
    criterion_ctx,
    criterion_text,
    tokenizer,
    device,
    lambda_reid=0.10,
    lambda_ground_mse=0.10,
    lambda_contrast=0.10,
    lambda_entity_pool=0.05,
    contrastive_tau=0.07,
    use_frame_aware_grounding=True,
    use_contrastive_roi=True,
    use_entity_pooling=True,
):
    model.train()
    running_loss = 0.0
    last_losses = {}

    for (frames, descriptions, image_target, text_target,
         roi1, roi2, roi_valid, roi_frame, ent_id) in dataloader:

        frames = frames.to(device)
        descriptions = descriptions.to(device)
        image_target = image_target.to(device)
        text_target = text_target.to(device)
        roi1 = roi1.to(device)
        roi2 = roi2.to(device)
        roi_valid = roi_valid.to(device)
        roi_frame = roi_frame.to(device)

        optimizer.zero_grad()

        pred_image_content, pred_image_context, predicted_text_logits_k, _, _, z_v_seq, z_t_seq = model(
            frames, descriptions, text_target
        )

        # Base losses
        loss_im = criterion_images(pred_image_content, image_target)
        mu_global = frames.mean(dim=[0, 1]).unsqueeze(0).expand_as(pred_image_context)
        loss_context = criterion_ctx(pred_image_context, mu_global)
        prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
        target_flat = text_target.squeeze(1)[:, 1:].reshape(-1)
        loss_text = criterion_text(prediction_flat, target_flat)

        # CoT grounding losses
        loss_reid = torch.tensor(0.0, device=device)
        loss_ground_mse = torch.tensor(0.0, device=device)
        loss_contrast = torch.tensor(0.0, device=device)
        loss_entity_pool = torch.tensor(0.0, device=device)

        if roi_valid.any():
            mask = roi_valid.bool()
            if mask.sum() > 0:
                z_r1 = model.image_encoder(roi1[mask])
                z_r2 = model.image_encoder(roi2[mask])
                loss_reid = F.mse_loss(z_r1, z_r2)

                if use_frame_aware_grounding:
                    f = roi_frame[mask].clamp(min=0, max=z_t_seq.size(1) - 1)
                    z_t_match = z_t_seq[mask].gather(
                        1, f.view(-1, 1, 1).expand(-1, 1, z_t_seq.size(-1))
                    ).squeeze(1)
                    loss_ground_mse = F.mse_loss(z_r1, z_t_match)

                    if use_contrastive_roi:
                        z_img_n = F.normalize(z_r1, dim=-1)
                        z_txt_n = F.normalize(z_t_match, dim=-1)
                        logits = (z_img_n @ z_txt_n.t()) / contrastive_tau
                        labels = torch.arange(logits.size(0), device=device)
                        loss_contrast = F.cross_entropy(logits, labels)

                if use_entity_pooling:
                    ent_list = [ent_id[i] for i, m in enumerate(mask.detach().cpu().tolist()) if m]
                    uniq = {}
                    for i_e, eid in enumerate(ent_list):
                        if eid:
                            uniq.setdefault(eid, []).append(i_e)
                    pool_losses = []
                    for eid, idxs in uniq.items():
                        if len(idxs) >= 2:
                            group = z_r1[idxs]
                            mean = group.mean(dim=0, keepdim=True)
                            pool_losses.append(F.mse_loss(group, mean.expand_as(group)))
                    if pool_losses:
                        loss_entity_pool = torch.stack(pool_losses).mean()

        loss = (
            loss_im + loss_context + loss_text
            + lambda_reid * loss_reid
            + lambda_ground_mse * loss_ground_mse
            + lambda_contrast * loss_contrast
            + lambda_entity_pool * loss_entity_pool
        )

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        last_losses = {
            'im': loss_im.item(),
            'ctx': loss_context.item(),
            'txt': loss_text.item(),
            'reid': float(loss_reid),
            'g_mse': float(loss_ground_mse),
            'nce': float(loss_contrast),
            'entpool': float(loss_entity_pool),
        }

    return running_loss / len(dataloader), last_losses


def run_training(
    model,
    train_dataloader,
    val_dataloader,
    tokenizer,
    device,
    n_epochs=5,
    lr=0.001,
    cot_config=None,
    val_fn=None,
):
    if cot_config is None:
        cot_config = {}

    criterion_images = nn.L1Loss()
    criterion_ctx = nn.MSELoss()
    criterion_text = nn.CrossEntropyLoss(
        ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    losses = []
    for epoch in range(n_epochs):
        epoch_loss, last = train_one_epoch(
            model, train_dataloader, optimizer,
            criterion_images, criterion_ctx, criterion_text,
            tokenizer, device, **cot_config,
        )
        losses.append(epoch_loss)
        print(
            f"Epoch [{epoch+1}/{n_epochs}] Loss: {epoch_loss:.4f}  "
            f"(im={last['im']:.3f}, ctx={last['ctx']:.3f}, txt={last['txt']:.3f}, "
            f"reid={last['reid']:.3f}, g_mse={last['g_mse']:.3f}, "
            f"nce={last['nce']:.3f}, entpool={last['entpool']:.3f})"
        )
        if val_fn is not None:
            val_fn(model, val_dataloader)
            model.train()

    return losses
