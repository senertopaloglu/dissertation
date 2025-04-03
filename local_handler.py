import os

import numpy as np

import torch

from sam2.build_sam import build_sam2_video_predictor

def run_segmentation(points, frame_idx, foldername, multi_resolution, is_first, is_final):
    """
    Runs segmentation on the specified video frames using the provided points.
    
    Parameters:
        points: A dictionary where keys are object IDs and values are lists of point coordinates and labels.
        frame_idx: The index of the reference frame for segmentation.
        foldername: The folder name containing video frames.
        multi_resolution: Boolean indicating if multi-resolution processing should be applied.
        is_first: Boolean flag indicating if this is the first segmentation invocation.
        is_final: Boolean flag indicating if this is the final segmentation (which triggers full video propagation).

    Returns:
        video_segments: A dictionary mapping frame indices to segmentation mask results.
    """
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    sam2_checkpoint = "./SAM_2_Medical_3D/sam2_hiera_large.pt"
    model_cfg = "./SAM_2_Medical_3D/sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    video_dir = f"./temp/{foldername}"
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    video_segments = {}  # video_segments contains the per-frame segmentation results

    prompts = {}  # hold all the clicks we add for visualization

    ann_frame_idx = frame_idx

    for k, v in points.items():
        ann_obj_id = k
        points = [[x[0], x[1]] for x in v]
        labels = [x[2] for x in v]
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, np.int32)
        prompts[ann_obj_id] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
    
    if is_final:
        # run propagation throughout the video and collect the results in a dict
        # prop forwards
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        # prop backwards
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx-1, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    else:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=0):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    
    return video_segments

def segment(slices, points, frame_idx, foldername, multi_resolution, is_first, is_final):
    """
    Invokes segmentation on frames using provided annotation points.
    This function is a wrapper that forwards the segmentation task to the
    local handler that runs the model locally.

    Parameters:
        slices: Unused parameter for segmentation slices.
        points: A dictionary where keys are object IDs and values are lists of point coordinates and labels.
        frame_idx: The index of the reference frame for segmentation.
        foldername: The folder name containing video frames.
        multi_resolution: Boolean indicating if multi-resolution processing should be applied.
        is_first: Boolean flag indicating if this is the first segmentation invocation.
        is_final: Boolean flag indicating if this is the final segmentation (which triggers full video propagation).

    Returns:
        video_segments: A dictionary mapping frame indices to segmentation mask results.
    """
    video_segments = run_segmentation(
        points,
        frame_idx,
        foldername,
        multi_resolution,
        is_first,
        is_final
    )
    return video_segments