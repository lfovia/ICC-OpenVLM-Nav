#!/usr/bin/env python3
"""


Same as your CLIP+DINO explore/servo script but with robust collision detection & recovery.
"""

import os
import csv
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import imageio
import torch
from transformers import CLIPProcessor, CLIPModel
import random
import time

import habitat
from config_utils import mp3d_config
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

# repo vision utils (must exist)
from cv_utils.detection_tools import initialize_dino_model, openset_detection
from cv_utils.segmentation_tools import initialize_sam_model

# ----------------------------
# Config / hyperparams
# ----------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_EPISODE_STEPS = 200
DIST_EPS = 0.02
DETECTION_STOP_AREA = 0.25
CENTER_OFFSET_THRESH = 0.12
CLIP_TEXT_PROMPTS = [
    "a photo of a {}",
    "an indoor photo of a {}",
    "a picture of a {}"
]

# collision/recovery params
STUCK_MOVE_THRESH = 0.03      # meters: if moved less than this, consider stuck after forward
STUCK_COUNT_TO_RECOVER = 2    # number of consecutive stuck detections before recovery
RECOVERY_MAX_TRIES = 3        # tries in recovery routine
RECOVERY_FORWARD_TRIES = 2    # forward tries when attempting to move away

# action ids (match your environment)
ACT_FORWARD = 1
ACT_TURN_LEFT = 2
ACT_TURN_RIGHT = 3
ACT_STOP = 0

# ----------------------------
# Initialize models
# ----------------------------
print("Device:", DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()

dino_model = None
sam_model = None

# ----------------------------
# Utilities
# ----------------------------
def safe_step(env, action, auto_reset=False):
    """Wrap env.step() and return observation dict or None if episode ended."""
    try:
        return env.step(action)
    except AssertionError:
        # Episode ended — optionally auto-reset
        print("[WARN] safe_step: Episode ended during step.")
        if auto_reset:
            print("[INFO] safe_step: auto-resetting environment.")
            return env.reset()
        return None

def adjust_topdown(metrics):
    try:
        img = colorize_draw_agent_and_fit_to_height(metrics["top_down_map"], 512)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.zeros((256, 256, 3), np.uint8)

def detect_conf_for_goal(image, category, dino_model, threshold=0.25):
    try:
        det_result = openset_detection(image, category, dino_model)
        if hasattr(det_result, "confidence") and det_result.confidence.size > 0:
            conf = float(np.max(det_result.confidence))
            return conf, det_result
        return 0.0, None
    except Exception as e:
        print("[WARN] DINO failed:", e)
        return 0.0, None

def bbox_area_ratio(bbox, image_shape):
    x1,y1,x2,y2 = map(int, bbox[:4])
    w = image_shape[1]; h = image_shape[0]
    area = max(0, (x2-x1)) * max(0, (y2-y1))
    return float(area) / (w*h + 1e-8)

def should_stop(bbox, img_shape, area_thresh=DETECTION_STOP_AREA):
    if bbox is None:
        return False
    return bbox_area_ratio(bbox, img_shape) >= area_thresh

# ----------------------------
# CLIP utilities
# ----------------------------
def encode_text_prompts(goal_category):
    prompts = [p.format(goal_category) for p in CLIP_TEXT_PROMPTS]
    inputs = clip_processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k,v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        text_feats = clip_model.get_text_features(**inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feat = text_feats.mean(dim=0, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat

def encode_image_clip(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k,v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        img_feat = clip_model.get_image_features(**inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    return img_feat

def clip_direction_reasoning(front_img, left_img, right_img, goal_category):
    text_feat = encode_text_prompts(goal_category)
    feats = [encode_image_clip(im) for im in (front_img, left_img, right_img)]
    scores = [float((f @ text_feat.T).item()) for f in feats]
    best_idx = int(np.argmax(scores))
    return best_idx, scores

# ----------------------------
# Local centering controller
# ----------------------------
def center_object_and_correct(env, rgb, bbox, center_thresh=CENTER_OFFSET_THRESH):
    if bbox is None:
        return rgb, False, None
    x1,y1,x2,y2 = bbox[:4]
    img_w = rgb.shape[1]
    cx = (x1 + x2) / 2.0
    offset = (cx - img_w/2.0) / (img_w/2.0)
    if abs(offset) <= center_thresh:
        return rgb, False, None
    # perform a small turn and return new rgb (using safe_step)
    if offset < 0:
        obs = safe_step(env, ACT_TURN_LEFT)
        if obs is None:
            return rgb, False, None
        return obs["rgb"], True, "left"
    else:
        obs = safe_step(env, ACT_TURN_RIGHT)
        if obs is None:
            return rgb, False, None
        return obs["rgb"], True, "right"

# ----------------------------
# Recovery routine
# ----------------------------
def recovery_backoff(env, tries=RECOVERY_MAX_TRIES):
    """
    Attempt to back off from collision:
      - rotate 180 (two right turns), try forward a couple times
      - rotate back
      - add a small random rotation to escape corners
    Returns True if recovery made a forward movement (approx), else False.
    """
    for t in range(tries):
        # rotate 180
        r = safe_step(env, 3)
        if r is None: return False
        r = safe_step(env, 3)
        if r is None: return False

        moved = False
        for _ in range(RECOVERY_FORWARD_TRIES):
            f = safe_step(env, 1)
            if f is None:
                return False
            # check approximate movement by reading agent pos if available
            # (caller will check exact movement)
            moved = True
            # append nothing here; caller of recovery can collect frames
        # rotate back (180)
        safe_step(env, 2)
        safe_step(env, 2)
        # small random rotation to avoid same trap
        if random.random() > 0.5:
            safe_step(env, 2)
        else:
            safe_step(env, 3)

        if moved:
            return True
    return False

# ----------------------------
# Main loop
# ----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="/tmp/zero_shot_clip_dirnav_recovery")
    p.add_argument("--max_steps", type=int, default=MAX_EPISODE_STEPS)
    return p.parse_args()

def zero_shot_nav_clip(eval_episodes=5, save_dir="/tmp/zero_shot_clip_dirnav", max_steps=200):
    os.makedirs(save_dir, exist_ok=True)
    habitat_cfg = mp3d_config(stage="val", episodes=eval_episodes)
    env = habitat.Env(habitat_cfg)

    global dino_model, sam_model
    dino_model = initialize_dino_model(device=DEVICE)
    try:
        sam_model = initialize_sam_model(device=DEVICE)
    except Exception:
        sam_model = None

    csv_path = os.path.join(save_dir, "clip_dirnav_metrics_with_recovery.csv")
    records = []

    for ep in tqdm(range(eval_episodes), desc="episodes"):
        obs = env.reset()
        if obs is None:
            obs = env.reset()
        goal = getattr(env.current_episode, "object_category", "unknown")
        print(f"\n[EP {ep}] goal = {goal}")

        ep_dir = os.path.join(save_dir, f"{goal}_traj_{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        rgb_writer = imageio.get_writer(os.path.join(ep_dir, "rgb.mp4"), fps=4)
        map_writer = imageio.get_writer(os.path.join(ep_dir, "map.mp4"), fps=4)

        episode_imgs = [obs["rgb"]]
        episode_maps = [adjust_topdown(env.get_metrics())]

        mode = "explore"
        no_dino_counter = 0
        prev_dist = env.get_metrics().get("distance_to_goal", np.inf)
        steps = 0
        reached = False

        # precompute text embedding
        text_feat = encode_text_prompts(goal)

        # position tracking for collision detection
        prev_pos = np.array(env.sim.get_agent_state().position)
        forward_stuck_counter = 0

        while (not env.episode_over) and steps < max_steps and not reached:
            # if episode ended mid-loop, break
            if env.episode_over:
                print(f"[EP {ep}] episode ended.")
                break

            rgb = obs["rgb"]

            # DINO detection
            d_conf, det_res = detect_conf_for_goal(rgb, goal, dino_model)
            det_exists = (det_res is not None and hasattr(det_res, "xyxy") and det_res.xyxy.size > 0)

            if det_exists:
                mode = "servo"
                no_dino_counter = 0
            else:
                if mode == "servo":
                    no_dino_counter += 1
                    if no_dino_counter > 8:
                        mode = "explore"
                        no_dino_counter = 0

            if mode == "explore":
                # sample front / left / right by rotating safely (use safe_step and check None)
                front = obs["rgb"]

                o = safe_step(env, 2)
                if o is None:
                    break
                left = o["rgb"]

                o = safe_step(env, 3)  # back to front
                if o is None:
                    break
                o = safe_step(env, 3)  # to right
                if o is None:
                    break
                right = o["rgb"]

                o = safe_step(env, 2)  # restore front
                if o is None:
                    break
                obs = o

                best_idx, scores = clip_direction_reasoning(front, left, right, goal)
                print(f"[EP {ep}][step {steps}] CLIP scores (front,left,right) = {scores} -> choose {best_idx}")

                # orient to chosen direction
                if best_idx == 1:
                    obs = safe_step(env, 2)
                    if obs is None: break
                elif best_idx == 2:
                    obs = safe_step(env, 3)
                    if obs is None: break

                # move forward a single step (then re-evaluate)
                prev_pos = np.array(env.sim.get_agent_state().position)
                obs = safe_step(env, 1)
                if obs is None: break
                episode_imgs.append(obs["rgb"])
                episode_maps.append(adjust_topdown(env.get_metrics()))
                steps += 1

                # collision detection by movement
                cur_pos = np.array(env.sim.get_agent_state().position)
                movement = np.linalg.norm(cur_pos - prev_pos)
                if movement < STUCK_MOVE_THRESH:
                    forward_stuck_counter += 1
                    print(f"[EP {ep}] small movement after forward: {movement:.3f} (stuck count {forward_stuck_counter})")
                else:
                    forward_stuck_counter = 0

                # if stuck repeatedly, run recovery
                if forward_stuck_counter >= STUCK_COUNT_TO_RECOVER:
                    print(f"[EP {ep}] Detected stuck during explore. Running recovery.")
                    recovered = recovery_backoff(env, tries=RECOVERY_MAX_TRIES)
                    if recovered:
                        
                        episode_imgs.append(obs["rgb"])
                        episode_maps.append(adjust_topdown(env.get_metrics()))
                        prev_pos = np.array(env.sim.get_agent_state().position)
                        forward_stuck_counter = 0
                        mode = "explore"
                        continue
                    else:
                        print(f"[EP {ep}] Recovery failed; continuing with explore.")
                        forward_stuck_counter = 0
                        continue

                # check immediate detection
                d_conf, det_res = detect_conf_for_goal(obs["rgb"], goal, dino_model)
                if det_res is not None and det_res.xyxy.size > 0:
                    mode = "servo"
                    continue

            elif mode == "servo":
                # visual servo
                # if no detection, try a small forward then re-evaluate
                d_conf, det_res = detect_conf_for_goal(rgb, goal, dino_model)
                if det_res is None or det_res.xyxy.size == 0:
                    no_dino_counter += 1
                    if no_dino_counter > 6:
                        mode = "explore"
                        no_dino_counter = 0
                        continue
                    prev_pos = np.array(env.sim.get_agent_state().position)
                    obs = safe_step(env, ACT_FORWARD)
                    if obs is None: break
                    episode_imgs.append(obs["rgb"])
                    episode_maps.append(adjust_topdown(env.get_metrics()))
                    steps += 1
                    cur_pos = np.array(env.sim.get_agent_state().position)
                    movement = np.linalg.norm(cur_pos - prev_pos)
                    if movement < STUCK_MOVE_THRESH:
                        # collision while trying to reacquire -> recovery
                        print(f"[EP {ep}] stuck while servo reacquire. Running recovery.")
                        recovered = recovery_backoff(env)
                        if recovered:
                            obs = env.get_observations()
                            episode_imgs.append(obs["rgb"])
                            episode_maps.append(adjust_topdown(env.get_metrics()))
                            prev_pos = np.array(env.sim.get_agent_state().position)
                            continue
                        else:
                            mode = "explore"
                            continue
                    continue

                # use best bbox
                idx = int(det_res.confidence.argmax())
                bbox = det_res.xyxy[idx][:4]

                # center if needed
                rgb_before = obs["rgb"]
                rgb_after, corrected, corr_action = center_object_and_correct(env, rgb_before, bbox, CENTER_OFFSET_THRESH)
                if corrected:
                    print(f"[EP {ep}][step {steps}] Correction: {corr_action}")
                    obs = env.get_observations()
                    episode_imgs.append(obs["rgb"])
                    episode_maps.append(adjust_topdown(env.get_metrics()))
                    steps += 1
                    # re-evaluate detection and bbox after correction
                    d_conf, det_res = detect_conf_for_goal(obs["rgb"], goal, dino_model)
                    if det_res is None or det_res.xyxy.size == 0:
                        # try small forward to re-acquire
                        prev_pos = np.array(env.sim.get_agent_state().position)
                        obs = safe_step(env, ACT_FORWARD)
                        if obs is None: break
                        episode_imgs.append(obs["rgb"])
                        episode_maps.append(adjust_topdown(env.get_metrics()))
                        steps += 1
                        cur_pos = np.array(env.sim.get_agent_state().position)
                        if np.linalg.norm(cur_pos - prev_pos) < STUCK_MOVE_THRESH:
                            # collision during approach -> recovery
                            print(f"[EP {ep}] collision during servo approach. Running recovery.")
                            recovered = recovery_backoff(env)
                            if recovered:
                                obs = env.get_observations()
                                episode_imgs.append(obs["rgb"])
                                episode_maps.append(adjust_topdown(env.get_metrics()))
                                prev_pos = np.array(env.sim.get_agent_state().position)
                                continue
                            else:
                                mode = "explore"
                                continue
                        continue
                    idx = int(det_res.confidence.argmax())
                    bbox = det_res.xyxy[idx][:4]

                # if centered enough, move forward
                if not should_stop(bbox, rgb_after, DETECTION_STOP_AREA):
                    prev_pos = np.array(env.sim.get_agent_state().position)
                    obs = safe_step(env, ACT_FORWARD)
                    if obs is None: break
                    episode_imgs.append(obs["rgb"])
                    episode_maps.append(adjust_topdown(env.get_metrics()))
                    steps += 1
                    cur_pos = np.array(env.sim.get_agent_state().position)
                    if np.linalg.norm(cur_pos - prev_pos) < STUCK_MOVE_THRESH:
                        print(f"[EP {ep}] stuck while moving toward centered object. Running recovery.")
                        recovered = recovery_backoff(env)
                        if recovered:
                            obs = env.get_observations()
                            episode_imgs.append(obs["rgb"])
                            episode_maps.append(adjust_topdown(env.get_metrics()))
                            prev_pos = np.array(env.sim.get_agent_state().position)
                            continue
                        else:
                            mode = "explore"
                            continue
                else:
                    print(f"[EP {ep}] Stop condition satisfied -> reached.")
                    # to register success in Habitat, usually STOP action is expected
                    _ = safe_step(env, ACT_STOP)
                    reached = True
                    break

            # stagnation check
            cur_dist = env.get_metrics().get("distance_to_goal", np.inf)
            if cur_dist >= prev_dist - DIST_EPS and not det_exists:
                # no meaningful progress; force explore
                mode = "explore"
            else:
                prev_dist = cur_dist

        # save frames
        for im, mp in zip(episode_imgs, episode_maps):
            try:
                rgb_writer.append_data(im)
                map_writer.append_data(mp)
            except Exception:
                pass
        rgb_writer.close()
        map_writer.close()

        # metrics
        m = env.get_metrics()
        rec = {
            "ep": ep,
            "success": 1 if reached or m.get("success", 0) else 0,
            "spl": m.get("spl", 0),
            "distance": m.get("distance_to_goal", float("inf")),
            "goal": goal
        }
        records.append(rec)
        with open(csv_path, "w", newline="") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        print(f"[EP {ep}] done. success={rec['success']}, distance={rec['distance']}")

    env.close()
    print("All episodes finished. Metrics in:", csv_path)


if __name__ == "__main__":
    args = get_args()
    zero_shot_nav_clip(eval_episodes=args.episodes, save_dir=args.save_dir, max_steps=args.max_steps)
