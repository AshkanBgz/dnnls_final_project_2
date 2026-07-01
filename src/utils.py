import re
import random
import textwrap
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from transformers import BertTokenizer


# ---------------------------------------------------------------------------
# Image / text helpers
# ---------------------------------------------------------------------------

def parse_gdi_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    images = []
    for gdi in soup.find_all('gdi'):
        image_id = None
        if gdi.attrs:
            for attr_name in gdi.attrs:
                if 'image' in attr_name.lower():
                    image_id = attr_name.replace('image', '')
                    break
        if not image_id:
            tag_str = str(gdi)
            match = re.search(r'<gdi\s+image(\d+)', tag_str)
            if match:
                image_id = match.group(1)
        if not image_id:
            image_id = str(len(images) + 1)

        content = gdi.get_text().strip()
        images.append({
            'image_id': image_id,
            'description': content,
            'objects': [o.get_text().strip() for o in gdi.find_all('gdo')],
            'actions': [a.get_text().strip() for a in gdi.find_all('gda')],
            'locations': [l.get_text().strip() for l in gdi.find_all('gdl')],
        })
    return images


def show_image(ax, image, de_normalize=False, img_mean=None, img_std=None):
    if de_normalize:
        image = transforms.Normalize(mean=-img_mean / img_std, std=1 / img_std)(image)
    ax.imshow(image.permute(1, 2, 0))


# ---------------------------------------------------------------------------
# CoT grounding helpers
# ---------------------------------------------------------------------------

def _parse_markdown_table(block: str) -> List[Dict[str, str]]:
    lines = [l.rstrip() for l in block.splitlines()]
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return []
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        if not line.strip().startswith("|"):
            break
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) == len(headers):
            rows.append(dict(zip(headers, cols)))
    return rows


def parse_cot_grounding(chain_of_thought: str) -> Dict[int, Dict[str, Any]]:
    frames: Dict[int, Dict[str, Any]] = {}
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought or ""))
    for i, m in enumerate(matches):
        img_idx = int(m.group(1)) - 1
        start = m.end()
        end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
        section = (chain_of_thought or "")[start:end]
        frames[img_idx] = {"characters": [], "objects": []}

        char_match = re.search(r"###\s*Characters(.*?)(?=\n###|\n##|$)", section, re.DOTALL)
        if char_match:
            for row in _parse_markdown_table(char_match.group(1)):
                cid, bbox_str = row.get("Character ID", "").strip(), row.get("Bounding Box", "").strip()
                if cid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["characters"].append({"id": cid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass

        obj_match = re.search(r"###\s*Objects(.*?)(?=\n###|\n##|$)", section, re.DOTALL)
        if obj_match:
            for row in _parse_markdown_table(obj_match.group(1)):
                oid, bbox_str = row.get("Object ID", "").strip(), row.get("Bounding Box", "").strip()
                if oid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["objects"].append({"id": oid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass
    return frames


def _clamp_bbox(x1, y1, x2, y2, W, H):
    x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
    y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_and_resize(pil_img, bbox, out_hw=(60, 125)):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, *pil_img.size)
    crop = pil_img.crop((x1, y1, x2, y2))
    crop = transforms.Resize(out_hw)(crop)
    return transforms.ToTensor()(crop)


def pick_reid_pair(frames_cot: Dict[int, Dict[str, Any]]) -> Optional[Tuple]:
    id_to_dets = {}
    for f_idx, content in frames_cot.items():
        for det in content.get("characters", []) + content.get("objects", []):
            ent_id = det.get("id")
            bbox = det.get("bbox")
            if ent_id and bbox:
                id_to_dets.setdefault(ent_id, []).append((f_idx, bbox))
    candidates = [k for k, v in id_to_dets.items() if len(v) >= 2]
    if not candidates:
        return None
    ent_id = random.choice(candidates)
    (f1, b1), (f2, b2) = random.sample(id_to_dets[ent_id], 2)
    return f1, f2, b1, b2, ent_id


def extract_cot_text_for_frame(chain_of_thought: str, frame_idx: int, max_chars: int = 600) -> str:
    if not chain_of_thought:
        return ""
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought))
    for i, m in enumerate(matches):
        if int(m.group(1)) - 1 == frame_idx:
            start = m.end()
            end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
            target = chain_of_thought[start:end]
            lines = [
                l for l in target.splitlines()
                if not l.strip().startswith("|") and not set(l.strip()) <= set("-|:")
            ]
            text = re.sub(r"\s+", " ", " ".join(l.strip() for l in lines if l.strip())).strip()
            return text[:max_chars]
    return ""


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SequencePredictionDataset(Dataset):
    def __init__(
        self,
        original_dataset,
        tokenizer,
        K: int = 4,
        max_len: int = 120,
        image_hw=(60, 125),
        use_cot_text: bool = True,
    ):
        super().__init__()
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.K = K
        self.max_len = max_len
        self.image_hw = image_hw
        self.use_cot_text = use_cot_text
        self.transform = transforms.Compose([
            transforms.Resize(image_hw),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames = self.dataset[idx]["images"]
        image_attributes = parse_gdi_text(self.dataset[idx]["story"])
        cot = self.dataset[idx].get("chain_of_thought", "")
        cot_frames = parse_cot_grounding(cot)

        frame_tensors, description_list = [], []
        for frame_idx in range(self.K):
            input_frame = self.transform(FT.equalize(frames[frame_idx]))
            frame_tensors.append(input_frame)

            description = image_attributes[frame_idx]["description"]
            if self.use_cot_text:
                cot_txt = extract_cot_text_for_frame(cot, frame_idx)
                if cot_txt:
                    description = description + " [COT] " + cot_txt

            input_ids = self.tokenizer(
                description,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
            ).input_ids.squeeze(0)
            description_list.append(input_ids)

        image_target = self.transform(FT.equalize(frames[self.K]))
        target_desc = image_attributes[self.K]["description"]
        target_ids = self.tokenizer(
            target_desc,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        ).input_ids

        roi_valid = torch.tensor(0, dtype=torch.long)
        roi1 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi2 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi_frame = torch.tensor(-1, dtype=torch.long)
        ent_id = ""

        pair = pick_reid_pair(cot_frames)
        if pair is not None:
            f1, f2, b1, b2, ent_id = pair
            if (0 <= f1 < self.K) and (0 <= f2 < self.K):
                try:
                    roi1 = crop_and_resize(frames[f1], b1, out_hw=self.image_hw)
                    roi2 = crop_and_resize(frames[f2], b2, out_hw=self.image_hw)
                    roi_valid = torch.tensor(1, dtype=torch.long)
                    roi_frame = torch.tensor(int(f1), dtype=torch.long)
                except Exception:
                    pass

        return (
            torch.stack(frame_tensors),
            torch.stack(description_list),
            image_target,
            target_ids,
            roi1, roi2, roi_valid, roi_frame, ent_id,
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)


def generate(decoder, hidden, cell, max_len, sos_token_id, eos_token_id, device='cpu'):
    decoder.eval()
    dec_input = torch.tensor([[sos_token_id]], dtype=torch.long, device=device)
    generated_tokens = []
    for _ in range(max_len):
        with torch.no_grad():
            prediction, hidden, cell = decoder(dec_input, hidden, cell)
        logits = prediction.squeeze(1)
        probs = torch.softmax(logits / 0.9, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.squeeze().item()
        if token_id == eos_token_id:
            break
        if token_id not in (0, sos_token_id):
            generated_tokens.append(token_id)
        dec_input = next_token
    return generated_tokens


def validation(model, data_loader, tokenizer, device):
    model.eval()
    with torch.no_grad():
        frames, descriptions, image_target, text_target, roi1, roi2, roi_valid, roi_frame, ent_id = next(iter(data_loader))
        frames = frames.to(device)
        descriptions = descriptions.to(device)
        image_target = image_target.to(device)
        text_target = text_target.to(device)

        predicted_image_k, context_image, _, hidden, cell, _, _ = model(frames, descriptions, text_target)

        fig, ax = plt.subplots(2, 6, figsize=(20, 5), gridspec_kw={'height_ratios': [2, 1.5]})
        for i in range(4):
            show_image(ax[0, i], frames[0, i].cpu())
            ax[0, i].axis('off')
            wrapped = textwrap.fill(tokenizer.decode(descriptions[0, i].cpu(), skip_special_tokens=True), width=40)
            ax[1, i].text(0.5, 0.99, wrapped, ha='center', va='top', fontsize=9)
            ax[1, i].axis('off')

        show_image(ax[0, 4], image_target[0].cpu())
        ax[0, 4].set_title('Target')
        ax[0, 4].axis('off')
        wrapped = textwrap.fill(tokenizer.decode(text_target.squeeze(1)[0].cpu(), skip_special_tokens=True), width=40)
        ax[1, 4].text(0.5, 0.99, wrapped, ha='center', va='top', fontsize=9)
        ax[1, 4].axis('off')

        show_image(ax[0, 5], context_image[0].cpu())
        ax[0, 5].set_title('Predicted')
        ax[0, 5].axis('off')
        gen_tokens = generate(
            model.text_decoder,
            hidden[:, 0, :].unsqueeze(1),
            cell[:, 0, :].unsqueeze(1),
            max_len=150,
            sos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            device=device,
        )
        wrapped = textwrap.fill(tokenizer.decode(gen_tokens), width=40)
        ax[1, 5].text(0.5, 0.99, wrapped, ha='center', va='top', fontsize=9)
        ax[1, 5].axis('off')
        plt.tight_layout()
        plt.show()
