#!/usr/bin/env python3
"""

-------------------------------------
Zero-shot Object Navigation using BLIP reasoning (replaces Qwen2-VL).

Pipeline:
1. Habitat environment (MP3D or HM3D)
2. BLIP for visual reasoning (zero-shot)
3. DINO for open-set object detection (optional)
4. SAM for segmentation (optional)
5. Simple step-based policy for navigation
6. Logs metrics + renders videos per episode

Author: Athira Krishnan R (2025) - adapted to BLIP
"""

import os
import csv
import argparse
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# BLIP imports (lightweight BLIP)
from transformers import BlipProcessor, BlipForConditionalGeneration

# Habitat imports
import habitat
from config_utils import mp3d_config  # or hm3d_config

# Vision tools
try:
    from cv_utils.detection_tools import initialize_dino_model, openset_detection
    from cv_utils.segmentation_tools import initialize_sam_model
except ImportError as e:
    print(f"[WARN] Could not import cv_utils modules: {e}")
    initialize_dino_model = openset_detection = initialize_sam_model = None

# ----------------------------
# Global setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ----------------------------
# Load BLIP model (replaces Qwen3-VL)
# ----------------------------
# Using a lightweight BLIP checkpoint so it fits into 11GB GPU in fp16.
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

print(f"[INFO] Loading BLIP model '{BLIP_MODEL_NAME}' (fp16 if CUDA)...")
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)

# Try fp16 device load; fallback to fp32 if not available
try:
    blip_model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
except Exception as e:
    # Some environments may not support device_map auto; load to cpu then move
    print(f"[WARN] device_map auto load failed: {e}. Loading normally and moving to device.")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    blip_model.to(device)

blip_model.eval()
print("[INFO] BLIP model ready.")

# ----------------------------
# Helper classes
# ----------------------------
class DummyPolicy(nn.Module):
    def forward(self, x):
        return torch.zeros(1, dtype=torch.float32)


# ----------------------------
# Utility functions (same as original)
# ----------------------------
def process_vision_info(messages):
    """Extract image and video inputs from messages (kept for compatibility)."""
    image_inputs = []
    video_inputs = []
    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "image":
                    image_inputs.append(item["image"])
                elif item.get("type") == "video":
                    video_inputs.append(item["video"])
    return image_inputs if image_inputs else None, video_inputs if video_inputs else None


def save_image_for_qwen(image_bgr, out_dir="./temp_blip_imgs"):
    """Save BGR image as PIL Image for BLIP input (kept name for compatibility)."""
    os.makedirs(out_dir, exist_ok=True)
    import time
    path = os.path.join(out_dir, f"img_{int(time.time()*1000)}.jpg")
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Debug: check image size and compute hash to verify images are changing
    if not hasattr(save_image_for_qwen, 'logged'):
        print(f"[DEBUG] Image shape: {image_bgr.shape}, PIL size: {pil_img.size}")
        save_image_for_qwen.logged = True

    # Quick hash to verify images are actually different
    import hashlib
    img_hash = hashlib.md5(image_bgr.tobytes()).hexdigest()[:8]
    if not hasattr(save_image_for_qwen, 'last_hash'):
        save_image_for_qwen.last_hash = None

    if save_image_for_qwen.last_hash != img_hash:
        print(f"[DEBUG] Image changed (hash: {img_hash})")
        save_image_for_qwen.last_hash = img_hash
    else:
        print(f"[DEBUG] WARNING: Same image as previous step!")

    pil_img.save(path)
    return pil_img


def adjust_topdown(metrics):
    """Convert Habitat top-down map to RGB."""
    try:
        from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
        img = colorize_draw_agent_and_fit_to_height(metrics["top_down_map"], 512)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.zeros((256, 256, 3), np.uint8)


def get_panorama(env, episode_images, episode_topdowns, steps=11):
    """Collect panoramic RGB frames."""
    imgs, depths = [], []
    for _ in range(steps):
        obs = env.step(3)  # turn right
        imgs.append(obs["rgb"])
        depths.append(obs.get("depth", None))
        episode_images.append(obs["rgb"])
        episode_topdowns.append(adjust_topdown(env.get_metrics()))
    return imgs, depths


def detect_object(image, category, dino_model, threshold=0.4):
    """Detect object bounding box with DINO."""
    if not dino_model or not openset_detection:
        return False, None
    try:
        det_result = openset_detection(image, category, dino_model)
        if hasattr(det_result, "xyxy") and det_result.xyxy.shape[0] > 0:
            conf = det_result.confidence
            valid = conf > threshold
            if valid.sum() == 0:
                return False, None
            idx = np.argmax(conf[valid])
            return True, det_result.xyxy[valid][idx]
    except Exception as e:
        print(f"[WARN] DINO detection failed: {e}")
    return False, None


def get_sam_mask(image, bbox, sam_model):
    """Generate segmentation mask from SAM."""
    if not sam_model:
        return np.zeros(image.shape[:2], np.uint8)
    try:
        x1, y1, x2, y2 = map(int, bbox[:4])
        sam_model.set_image(np.array(image))
        box = np.array([[x1, y1, x2, y2]])
        masks, _, _ = sam_model.predict(box=box)
        return masks[0].astype(np.uint8) * 255
    except Exception as e:
        print(f"[WARN] SAM segmentation failed: {e}")
        return np.zeros(image.shape[:2], np.uint8)


# ----------------------------
# BLIP Reasoning Functions (replace Qwen functions)
# ----------------------------

def blip_generate_caption(image, goal):
    """Generate a detailed caption describing the scene using BLIP."""
    img_pil = save_image_for_qwen(image)

    prompt = (
        "Describe this scene in detail as if you are an agent navigating indoors. "
        "Describe what you can see directly ahead (forward), what might be visible to your left, and to your right. "
        "Mention any doors, hallways, furniture, objects, walls, or open spaces. Be specific in 2-3 sentences."
    )
    if goal:
        prompt = f"You are searching for a {goal}. " + prompt

    # Prepare inputs: processor(images=..., text=...) supports BLIP conditional generation
    inputs = blip_processor(images=img_pil, text=prompt, return_tensors="pt").to(device)

    # Run generate in autocast if CUDA available to use fp16
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = blip_model.generate(**inputs, max_new_tokens=128)
        else:
            out = blip_model.generate(**inputs, max_new_tokens=128)

    # Decode
    try:
        caption = blip_processor.tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        caption = out[0].cpu().numpy().tolist()
        caption = str(caption)

    caption = caption.strip()
    print(f"[DEBUG] Scene caption (BLIP): {caption}")
    return caption


def blip_reason_action(image, goal, step_num=0):
    """Two-stage reasoning: generate caption then ask BLIP to pick an action."""
    # Stage 1: caption
    caption = blip_generate_caption(image, goal)

    # Stage 2: ask BLIP to pick the best action given caption + image
    img_pil = save_image_for_qwen(image)

    action_prompt = (
        f"You are navigating indoors to find a {goal}.\n\n"
        f"Scene description:\n{caption}\n\n"
        "Based on this description and the image, choose the BEST navigation action:\n"
        "- FORWARD: if there is a clear path, hallway, corridor, or open space ahead\n"
        "- LEFT: if there is a door, doorway, alley, or interesting opening to the left\n"
        "- RIGHT: if there is a door, doorway, alley, or interesting opening to the right\n"
        f"- STOP: ONLY if you can clearly see the {goal} object directly in front and very close\n\n"
        "Reply with exactly ONE WORD: FORWARD, LEFT, RIGHT, or STOP"
    )

    inputs = blip_processor(images=img_pil, text=action_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = blip_model.generate(**inputs, max_new_tokens=32)
        else:
            out = blip_model.generate(**inputs, max_new_tokens=32)

    try:
        decoded = blip_processor.tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        decoded = out[0].cpu().numpy().tolist()
        decoded = str(decoded)

    print(f"[DEBUG] Full BLIP response: '{decoded}'")

    answer = decoded.strip().lower()
    print(f"[DEBUG] BLIP action response: '{answer}'")

    # Prevent premature stopping
    if "stop" in answer and step_num < 5:
        print(f"[DEBUG] Ignoring STOP at step {step_num} (too early) - defaulting to FORWARD")
        return 1
    elif "stop" in answer:
        print(f"[DEBUG] BLIP says STOP - ending navigation")
        return 0
    elif "forward" in answer:
        return 1
    elif "left" in answer:
        return 2
    elif "right" in answer:
        return 3
    else:
        # fallback: simple heuristic on caption text (if BLIP didn't produce one-word reply)
        c = caption.lower()
        if "left" in c and "right" not in c:
            return 2
        if "right" in c and "left" not in c:
            return 3
        if any(k in c for k in ["corridor", "hallway", "ahead", "open space", "path"]):
            return 1
        print(f"[DEBUG] No clear action keyword found in '{answer}', defaulting to FORWARD")
        return 1


# ----------------------------
# Main pipeline (uses BLIP reasoner)
# ----------------------------
def run_zero_shot(eval_episodes=10):
    """Run BLIP zero-shot navigation episodes."""
    habitat_config = mp3d_config(stage="val", episodes=eval_episodes)
    env = habitat.Env(habitat_config)

    print("[INFO] Initializing models...")
    dino_model = initialize_dino_model(device=device) if initialize_dino_model else None
    sam_model = initialize_sam_model(device=device) if initialize_sam_model else None

    metrics_all = []

    for ep in tqdm(range(eval_episodes), desc="Episodes"):
        obs = env.reset()
        goal = getattr(env.current_episode, "object_category", "object")

        print(f"[EP {ep}] Goal: {goal}")
        os.makedirs(f"./results_blip/{goal}", exist_ok=True)
        dir_out = f"./results_blip/{goal}/traj_{ep}"
        os.makedirs(dir_out, exist_ok=True)

        writer_rgb = imageio.get_writer(f"{dir_out}/rgb.mp4", fps=4)
        writer_top = imageio.get_writer(f"{dir_out}/map.mp4", fps=4)

        imgs, tops = [obs["rgb"]], [adjust_topdown(env.get_metrics())]

        # Start directly from current view (no initial panorama sweep)

        import hashlib
        stuck_history = []
        stuck_threshold = 5  # if same hash appears 5 times, agent is stuck

        t = 0
        while not env.episode_over and t < 100:
            img = obs["rgb"]

            # Compute hash to detect stuck state
            img_hash = hashlib.md5(img.tobytes()).hexdigest()[:8]
            stuck_history.append(img_hash)
            if len(stuck_history) > 10:
                stuck_history.pop(0)

            # Check if stuck (same view repeated too many times)
            if stuck_history.count(img_hash) >= stuck_threshold:
                print(f"[EP {ep}] t={t}, STUCK DETECTED (hash {img_hash} repeated) - Taking corrective turn")
                import random
                corrective_action = random.choice([2, 3])  # random turn left or right
                obs = env.step(corrective_action)
                imgs.append(obs["rgb"])
                tops.append(adjust_topdown(env.get_metrics()))
                stuck_history.clear()  # reset stuck history after correction
                t += 1
                continue

            # Ask BLIP for action from current view
            act = blip_reason_action(img, goal, step_num=t)
            action_map = {0: 0, 1: 1, 2: 2, 3: 3}  # stop, fwd, left, right

            # If BLIP decided to stop, break the loop
            if act == 0:
                print(f"[EP {ep}] t={t}, action={act} (STOP) - Breaking loop")
                obs = env.step(action_map.get(act, 0))
                imgs.append(obs["rgb"])
                tops.append(adjust_topdown(env.get_metrics()))
                break

            obs = env.step(action_map.get(act, 1))
            action_names = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
            print(f"[EP {ep}] t={t}, action={action_names.get(act, act)}")
            imgs.append(obs["rgb"])
            tops.append(adjust_topdown(env.get_metrics()))
            t += 1

        metrics = env.get_metrics()
        metrics_all.append({
            "ep": ep,
            "success": metrics.get("success", 0),
            "spl": metrics.get("spl", 0),
            "distance": metrics.get("distance_to_goal", 999),
            "goal": goal
        })

        for i, (im, top) in enumerate(zip(imgs, tops)):
            writer_rgb.append_data(im)
            writer_top.append_data(top)
        writer_rgb.close()
        writer_top.close()

        # Save metrics
        with open("blip_objnav_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_all[0].keys())
            writer.writeheader()
            writer.writerows(metrics_all)

        print(f"[EP {ep}] Done. Metrics saved.")

    env.close()
    print("[INFO] All episodes completed.")


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    run_zero_shot(args.episodes)
