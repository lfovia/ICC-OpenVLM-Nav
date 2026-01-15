#!/usr/bin/env python3
"""
zero_shot_clip_phi3_objnav.py
-------------------------------------
Zero-shot Object Navigation using CLIP for perception + Phi-3-mini for reasoning.
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
import clip
from transformers import AutoTokenizer, AutoModelForCausalLM
import habitat
from config_utils import mp3d_config

# ----------------------------
# Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# Load CLIP
print("[INFO] Loading CLIP (ViT-B/32)...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Load Phi-3-mini reasoning model
print("[INFO] Loading Phi-3-mini...")
phi_name = "microsoft/Phi-3-mini-4k-instruct"
phi_tokenizer = AutoTokenizer.from_pretrained(phi_name)
phi_model = AutoModelForCausalLM.from_pretrained(
    phi_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
phi_model.eval()

# ----------------------------
# Helpers
# ----------------------------
def get_clip_similarity(image_bgr, goal_text):
    """Compute CLIP similarity between image and goal text."""
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
    text_input = clip.tokenize([f"a {goal_text}"]).to(device)
    with torch.no_grad():
        image_feat = clip_model.encode_image(image_input)
        text_feat = clip_model.encode_text(text_input)
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        sim = (image_feat @ text_feat.T).item()
    return sim


def reason_action_phi3(sim, goal, step_num=0):
    """Text-only reasoning given CLIP score + goal."""
    prompt = (
        f"You are an indoor navigation robot searching for a {goal}.\n"
        f"Current similarity to goal object (0–1): {sim:.2f}\n"
        "If similarity > 0.30, the object is probably in view.\n\n"
        "Decide ONE best navigation action:\n"
        "FORWARD - move ahead if you see open space or the goal seems farther.\n"
        "LEFT - turn left if you suspect the goal might be on the left.\n"
        "RIGHT - turn right if it might be on the right.\n"
        "STOP - only if the object is clearly visible and close.\n"
        "Respond with exactly one of: FORWARD, LEFT, RIGHT, STOP."
    )
    inputs = phi_tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    with torch.no_grad():
        out = phi_model.generate(**inputs, max_new_tokens=32)
    decoded = phi_tokenizer.decode(out[0], skip_special_tokens=True)
    answer = decoded.split(prompt)[-1].strip().upper()
    print(f"[PHI3] {answer}")

    if "STOP" in answer and step_num < 5:
        print(f"[DEBUG] Early STOP ignored")
        return 1
    if "STOP" in answer:
        return 0
    if "LEFT" in answer:
        return 2
    if "RIGHT" in answer:
        return 3
    return 1  # default FORWARD


def adjust_topdown(metrics):
    try:
        from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
        img = colorize_draw_agent_and_fit_to_height(metrics["top_down_map"], 512)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.zeros((256, 256, 3), np.uint8)


# ----------------------------
# Main loop
# ----------------------------
def run_zero_shot(eval_episodes=5):
    cfg = mp3d_config(stage="val", episodes=eval_episodes)
    env = habitat.Env(cfg)
    metrics_all = []

    for ep in tqdm(range(eval_episodes), desc="Episodes"):
        obs = env.reset()
        goal = getattr(env.current_episode, "object_category", "object")
        print(f"\n[EP {ep}] Goal: {goal}")

        os.makedirs(f"./results_clip_phi3/{goal}", exist_ok=True)
        dir_out = f"./results_clip_phi3/{goal}/traj_{ep}"
        os.makedirs(dir_out, exist_ok=True)
        writer_rgb = imageio.get_writer(f"{dir_out}/rgb.mp4", fps=4)
        writer_top = imageio.get_writer(f"{dir_out}/map.mp4", fps=4)

        imgs, tops = [obs["rgb"]], [adjust_topdown(env.get_metrics())]

        t = 0
        while not env.episode_over and t < 100:
            img = obs["rgb"]
            sim = get_clip_similarity(img, goal)
            print(f"[CLIP] Similarity={sim:.3f}")

            if sim > 0.32:
                print(f"[EP {ep}] STOP condition met (sim={sim:.2f})")
                act = 0
            else:
                act = reason_action_phi3(sim, goal, step_num=t)

            obs = env.step(act)
            imgs.append(obs["rgb"])
            tops.append(adjust_topdown(env.get_metrics()))
            names = {0:"STOP",1:"FORWARD",2:"LEFT",3:"RIGHT"}
            print(f"[EP {ep}] Step {t}: {names.get(act)}")

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

        with open("clip_phi3_objnav_metrics_50.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_all[0].keys())
            writer.writeheader()
            writer.writerows(metrics_all)

        print(f"[EP {ep}] Done. Metrics saved.")
    env.close()


# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    run_zero_shot(args.episodes)
