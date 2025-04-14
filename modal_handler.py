"""
Module for handling remote segmentation of 3d volumes.
"""
import os
import sys
import site

import torch
import numpy as np
import matplotlib.pyplot as plt

import modal

from type_aliases import Points, SegmentationResult

cuda_version = "12.2.2"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel"
    ).run_commands(
        "python --version",
        "apt install -y clang",
        "clang --version",
        "pip install --upgrade pip",
        "pip install --upgrade setuptools",
        "pip install matplotlib",
        "CC=gcc CXX=g++",
        "pip install torch torchvision torchaudio --no-build-isolation --index-url https://download.pytorch.org/whl/cu121",
        "pip install tqdm"
    )
)
vol = modal.Volume.from_name("my_adapted_sam_2_medical_3d")

app = modal.App("adapted-example-3")


@app.function(gpu="L4", image=image, volumes={"/root/temp":vol}, timeout=1000, mounts=[])
def run_segmentation(
    points_np: Points,
    frame_idx: int,
    foldername: str,
    multi_resolution: bool,
    is_first: bool,
    is_final: bool,
    is_global: bool
) -> SegmentationResult:
    import subprocess
    
    print("#" * 30)
    print(f"points : {points_np}")
    print("#" * 30)

    os.chdir(os.path.expanduser("temp/SAM_2_Medical_3D"))

    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        subprocess.call([sys.executable, "-m", "venv", venv_dir])
    venv_python = os.path.join(venv_dir, "bin", "python")
    venv_site = os.path.join(venv_dir, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    
    # Update current process to use venv's site-packages
    if venv_site not in sys.path:
        site.addsitedir(venv_site)

    if not multi_resolution or is_first:
        try:
            subprocess.check_output([venv_python, '-m', 'pip', 'show', 'hydra-core'])
        except subprocess.CalledProcessError as e:
            subprocess.call([venv_python, '-m', 'pip', 'install', 'hydra-core'])

        try:
            subprocess.check_output([venv_python, '-m', 'pip', 'show', "SAM-2-For-Medical-3D"])
        except:
            print("did not find -e .[demo] i will install now")
            subprocess.call([venv_python, '-m', 'pip', 'install', '-e', ".[demo]"])

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    current_directory = os.getcwd()
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + current_directory 
    if current_directory not in sys.path:
        sys.path.append(current_directory)
    if multi_resolution and is_first:
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + current_directory 

        if current_directory not in sys.path:
            sys.path.append(current_directory)

    from sam2.build_sam import build_sam2_video_predictor
    sam2_checkpoint = "./sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    video_dir = f"./frames/{foldername}"

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)


    volume_segments = {}  # contains the per-frame segmentation results

    prompts = {}  # contains all the clicks we add for visualisation

    if not is_global:
        ann_frame_idx = frame_idx  # the frame index of the annotation
    else:
        ann_frame_idx = None

    for k,v in points_np.items():
        ann_obj_id = k
        points = []
        labels = []
        if is_global:
            for f_index, v in v.items():
                if ann_frame_idx is None:
                    ann_frame_idx = f_index
                else:
                    ann_frame_idx = min(ann_frame_idx, f_index)
                points.extend([[x[0], x[1]] for x in v])
                labels.extend([x[2] for x in v])
                points_np = np.array(points, dtype=np.float32)
                labels_np = np.array(labels, np.int32)
                prompts[ann_obj_id] = points_np, labels_np
                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=f_index,
                    obj_id=ann_obj_id,
                    points=points_np,
                    labels=labels_np,
                )
        else:
            points = [[x[0], x[1]] for x in v]
            labels = [x[2] for x in v]
            points_np = np.array(points, dtype=np.float32)
            labels_np = np.array(labels, np.int32)
            prompts[ann_obj_id] = points_np, labels_np
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_np,
                labels=labels_np,
            )

    if is_final:
        # run propagation throughout the volume and collect the results in a dict
        # prop forwards
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx):
            volume_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        # prop backwards
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx-1, reverse=True):
            volume_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    else:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=0):
            volume_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    
    if "VIRTUAL_ENV" in os.environ:
        del os.environ["VIRTUAL_ENV"]

    return volume_segments

def segment(
    slices: np.ndarray,
    points: Points,
    frame_idx: int,
    foldername: str,
    multi_resolution: bool,
    is_first: bool,
    is_final: bool,
    is_global: bool = False,
) -> SegmentationResult:
    """
    Invoke remote segmentation on volume slices using provided annotation points.

    This function is a wrapper that forwards the segmentation task to the 
    remote function 'run_segmentation' via Modal (modal.com). The actual segmentation logic 
    is executed remotely, and results are returned as a dictionary mapping frame 
    indices to segmentation mask results.

    Args:
        slices: Unused parameter for segmentation slices.
        points: A dictionary where keys are object IDs and values are lists of point coordinates and labels.
        frame_idx: The index of the reference frame for segmentation.
        foldername: The folder name containing volume slices.
        multi_resolution: Boolean indicating if multi-resolution processing should be applied.
        is_first: Boolean flag indicating if this is the first segmentation invocation.
        is_final: Boolean flag indicating if this is the final segmentation (which triggers full volume propagation).

    Returns:
        volume_segments: A dictionary mapping frame indices to segmentation mask results.
    """
    with modal.enable_output():
        with app.run():
            volume_segments=run_segmentation.remote(points, frame_idx, foldername, multi_resolution, is_first, is_final, is_global)
    return volume_segments