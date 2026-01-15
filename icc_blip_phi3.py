#!/usr/bin/env python3
"""
zero_shot_blip_phi3_objnav.py
-------------------------------------
Zero-shot Object Navigation using BLIP for perception + Phi-3-mini for reasoning.
Lightweight alternative to Qwen2-VL.

Author: Athira Krishnan R (2025)
"""

import os
import csv
import argparse
import cv2
import imageio
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoModelForCausalLM, AutoTokenizer
)
import habitat
from config_utils import mp3d_config

# ----------------------------
# Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ----------------------------
# Load BLIP captioning model
# ----------------------------
print("[INFO] Loading BLIP model...")
blip_model_name = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(
    blip_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
blip_model.eval()

# ----------------------------
# Load Phi-3-mini reasoning model
# ----------------------------
print("[INFO] Loading Phi-3-mini reasoning model...")
llm_name = "microsoft/Phi-3-mini-4k-instruct"
phi_tokenizer = AutoTokenizer.from_pretrained(llm_name)
phi_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
phi_model.eval()

# ----------------------------
# Utility functions
# ----------------------------
def generate_blip_caption(image_bgr):
    """Generate caption using BLIP."""
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    print(f"[BLIP] {caption}")
    return caption


def reason_action_phi3(caption, goal, step_num=0):
    """Use Phi-3-mini to reason next action."""
    prompt = (
        f"You are an indoor navigation agent searching for a {goal}.\n"
        f"Scene description: {caption}\n\n"
        "Decide ONE of the following actions:\n"
        "FORWARD - if the way ahead looks open or has a doorway/hallway\n"
        "LEFT - if the {goal} or an open doorway appears on the left\n"
        "RIGHT - if the {goal} or an open doorway appears on the right\n"
        "STOP - only if the {goal} is clearly visible and very close.\n"
        "Reply with exactly one of: FORWARD, LEFT, RIGHT, STOP."
    )
    inputs = phi_tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    with torch.no_grad():
        out = phi_model.generate(**inputs, max_new_tokens=32)
    decoded = phi_tokenizer.decode(out[0], skip_special_tokens=True)
    answer = decoded.split(prompt)[-1].strip().upper()
    print(f"[PHI3] Response: {answer}")

    # Action mapping
    if "STOP" in answer and step_num < 5:
        print(f"[DEBUG] Early STOP ignored (step={step_num}) → FORWARD")
        return 1
    if "STOP" in answer:
        return 0
    if "FORWARD" in answer:
        return 1
    if "LEFT" in answer:
        return 2
    if "RIGHT" in answer:
        return 3
    print(f"[DEBUG] No clear action → FORWARD")
    return 1


def adjust_topdown(metrics):
    try:
        from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
        img = colorize_draw_agent_and_fit_to_height(metrics["top_down_map"], 512)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.zeros((256, 256, 3), np.uint8)


# ----------------------------
# Main pipeline
# ----------------------------
def run_zero_shot(eval_episodes=5):
    habitat_config = mp3d_config(stage="val", episodes=eval_episodes)
    env = habitat.Env(habitat_config)
    metrics_all = []

    for ep in tqdm(range(eval_episodes), desc="Episodes"):
        obs = env.reset()
        goal = getattr(env.current_episode, "object_category", "object")

        print(f"\n[EP {ep}] Goal: {goal}")
        os.makedirs(f"./results_blip_phi3/{goal}", exist_ok=True)
        dir_out = f"./results_blip_phi3/{goal}/traj_{ep}"
        os.makedirs(dir_out, exist_ok=True)

        writer_rgb = imageio.get_writer(f"{dir_out}/rgb.mp4", fps=4)
        writer_top = imageio.get_writer(f"{dir_out}/map.mp4", fps=4)

        imgs, tops = [obs["rgb"]], [adjust_topdown(env.get_metrics())]
        t = 0
        while not env.episode_over and t < 100:
            img = obs["rgb"]
            caption = generate_blip_caption(img)
            act = reason_action_phi3(caption, goal, step_num=t)
            action_map = {0: 0, 1: 1, 2: 2, 3: 3}
            obs = env.step(action_map.get(act, 1))
            imgs.append(obs["rgb"])
            tops.append(adjust_topdown(env.get_metrics()))
            action_names = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
            print(f"[EP {ep}] Step {t}: Action → {action_names.get(act)}")
            if act == 0:
                break
            t += 1

        metrics = env.get_metrics()
        metrics_all.append({
            "ep": ep,
            "success": metrics.get("success", 0),
            "spl": metrics.get("spl", 0),
            "distance": metrics.get("distance_to_goal", 999),
            "goal": goal
        })

        for im, top in zip(imgs, tops):
            writer_rgb.append_data(im)
            writer_top.append_data(top)
        writer_rgb.close()
        writer_top.close()

        with open("blip_phi3_objnav_metrics_50.csv", "w", newline="") as f:
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
