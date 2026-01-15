#!/usr/bin/env python3
"""
zero_shot_qwen_objnav_with_history.py
-------------------------------------
Zero-shot Object Navigation using Qwen-VL reasoning with short history context.

Pipeline:
1. Habitat environment (MP3D or HM3D)
2. Qwen-VL for visual reasoning (zero-shot) with short rolling history
3. DINO for open-set object detection (optional)
4. SAM for segmentation (optional)
5. Simple step-based policy for navigation
6. Logs metrics + renders videos per episode

Author: Athira Krishnan R (2025), Swapnil Bag
Adapted: added short history context for improved reasoning
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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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

# Load Qwen3-VL model (works with Python 3.9+)
model_name = "Qwen/Qwen3-VL-2B-Instruct"
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
qwen_model.eval()

# ----------------------------
# Helper classes
# ----------------------------
class DummyPolicy(nn.Module):
    def forward(self, x):
        return torch.zeros(1, dtype=torch.float32)

# ----------------------------
# Utility functions
# ----------------------------
def process_vision_info(messages):
    """Extract image and video inputs from messages for Qwen2-VL."""
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


def save_image_for_qwen(image_bgr, out_dir="./temp_qwen_imgs"):
    """Save BGR image as PIL Image for Qwen input."""
    os.makedirs(out_dir, exist_ok=True)
    import time
    path = os.path.join(out_dir, f"img_{int(time.time()*1000)}.jpg")
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Debug: check image size and compute hash to verify images are changing
    if not hasattr(save_image_for_qwen, 'logged'):
        try:
            print(f"[DEBUG] Image shape: {image_bgr.shape}, PIL size: {pil_img.size}")
        except Exception:
            pass
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
# Qwen Reasoning Functions (now with history)
# ----------------------------

def qwen_generate_caption(image, goal):
    """Generate a detailed caption describing the scene in all directions."""
    img_pil = save_image_for_qwen(image)
    
    prompt = (
        f"Describe this scene in detail as if you are an agent navigating indoors. "
        "Describe what you can see directly ahead (forward), what might be visible to your left, and to your right. "
        "Mention any doors, doorways, hallways, corridors, furniture, objects, walls, or open spaces. "
        "Be specific and detailed in 2-3 sentences."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text", "text": prompt}
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(qwen_model.device)
    
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=128)
    
    decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    if "assistant" in decoded.lower():
        caption = decoded.split("assistant")[-1].strip()
    else:
        caption = decoded.strip()
    
    print(f"[DEBUG] Scene caption: {caption}")
    return caption


def qwen_reason_action(image, goal, step_num=0, history=None):
    """
    Two-stage reasoning: generate caption first, then decide action based on
    caption + short history (history is a list of dicts: {'action':str, 'caption':str}).
    Returns (action_int, caption_str).
    """
    # Stage 1: Generate detailed scene caption
    caption = qwen_generate_caption(image, goal)
    
    # Format history (keep it concise)
    history_str = ""
    if history:
        # only show last 3 entries to keep prompt short
        for entry in history[-3:]:
            a = entry.get("action", "")
            c = entry.get("caption", "")
            history_str += f"Previous action: {a}. Previous observation: {c}\n"
        history_str = "Recent history:\n" + history_str + "\n"

    # Stage 2: Use caption + image + goal + history to decide action
    img_pil = save_image_for_qwen(image)
    
    prompt = (
        f"You are navigating indoors to find a {goal}.\n\n"
        f"{history_str}"
        f"Current scene description:\n{caption}\n\n"
        f"Based on this description and the image, choose the BEST navigation action:\n"
        f"- FORWARD: if there is a clear path, hallway, corridor, or open space ahead\n"
        f"- LEFT: if there is a door, doorway, alley, or interesting opening to the left\n"
        f"- RIGHT: if there is a door, doorway, alley, or interesting opening to the right\n"
        f"- STOP: ONLY if you can clearly see the {goal} object directly in front and very close\n\n"
        f"Exploration strategy:\n"
        f"- Prefer exploring doorways, corridors, and open spaces\n"
        f"Be concise and reply with exactly ONE WORD: FORWARD, LEFT, RIGHT, or STOP"
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text", "text": prompt}
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(qwen_model.device)
    
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=32)
    
    decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"[DEBUG] Full Qwen response: '{decoded}'")
    
    if "assistant" in decoded.lower():
        answer = decoded.lower().split("assistant")[-1].strip()
    else:
        answer = decoded.strip().lower()
    
    print(f"[DEBUG] Qwen action response: '{answer}'")
    
    # Prevent premature stopping
    if "stop" in answer and step_num < 5:
        print(f"[DEBUG] Ignoring STOP at step {step_num} (too early) - defaulting to FORWARD")
        return 1, caption
    elif "stop" in answer:
        print(f"[DEBUG] Qwen says STOP - ending navigation")
        return 0, caption
    elif "forward" in answer:
        return 1, caption
    elif "left" in answer:
        return 2, caption
    elif "right" in answer:
        return 3, caption
    else:
        print(f"[DEBUG] No clear action keyword found in '{answer}', defaulting to FORWARD")
        return 1, caption


# ----------------------------
# Main pipeline
# ----------------------------
def run_zero_shot(eval_episodes=10):
    """Run Qwen zero-shot navigation episodes (with short history context)."""
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
        os.makedirs(f"./results_qwen/{goal}", exist_ok=True)
        dir_out = f"./results_qwen/{goal}/traj_{ep}"
        os.makedirs(dir_out, exist_ok=True)

        writer_rgb = imageio.get_writer(f"{dir_out}/rgb.mp4", fps=4)
        writer_top = imageio.get_writer(f"{dir_out}/map.mp4", fps=4)

        imgs, tops = [obs["rgb"]], [adjust_topdown(env.get_metrics())]

        # history keeps last N (action, caption)
        history = []
        max_history_len = 5

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
            
            # Ask Qwen for action from current view, passing short history
            act, caption = qwen_reason_action(img, goal, step_num=t, history=history)
            action_map = {0: 0, 1: 1, 2: 2, 3: 3}  # stop, fwd, left, right
            
            # record history (convert action int to readable string)
            action_names = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
            history.append({"action": action_names.get(act, str(act)), "caption": caption})
            # cap history size
            if len(history) > max_history_len:
                history.pop(0)
            
            # If Qwen decided to stop, perform stop step then break the loop
            if act == 0:
                print(f"[EP {ep}] t={t}, action={act} (STOP) - Breaking loop")
                obs = env.step(action_map.get(act, 0))
                imgs.append(obs["rgb"])
                tops.append(adjust_topdown(env.get_metrics()))
                break
            
            obs = env.step(action_map.get(act, 1))
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
        if metrics_all:
            with open("qwen_objnav_metrics_history_50v2.csv", "w", newline="") as f:
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
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    run_zero_shot(args.episodes)
