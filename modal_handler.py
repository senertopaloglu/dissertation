import os
import sys
import site

import torch
import numpy as np
import matplotlib.pyplot as plt

import modal

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
        "pip install torch torchvision torchaudio --no-build-isolation --index-url https://download.pytorch.org/whl/cu121"
    )
)
vol = modal.Volume.from_name("sam_2_medical_3d")

app = modal.App("example-3")



def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

@app.function(gpu="L4", image=image, volumes={"/root/temp":vol}, timeout=1000, mounts=[])
def do_some_magic(points, frame_idx, foldername, multi_resolution, is_first, is_final):
    import subprocess
    
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
            subprocess.call([venv_python, '-m', 'pip', 'show', ".[demo]"])
        except subprocess.CalledProcessError as e:
            subprocess.call([venv_python, '-m', 'pip', 'install', '--prefer-binary', '--no-build-isolation', "-e", ".[demo]"])    

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    current_directory = os.getcwd()
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


    video_segments = {}  # video_segments contains the per-frame segmentation results

    prompts = {}  # hold all the clicks we add for visualization

    ann_frame_idx = frame_idx  # the frame index we interact with TODO: set = frame_idx (parameter)

    for k,v in points.items():
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
    
    if "VIRTUAL_ENV" in os.environ:
        del os.environ["VIRTUAL_ENV"]

    return video_segments

def segment(slices, points, frame_idx, foldername, multi_resolution, is_first, is_final):
    """
    Invoke remote segmentation on video frames using provided annotation points.

    This function is a wrapper that forwards the segmentation task to the 
    remote function 'do_some_magic' via Modal (modal.com). The actual segmentation logic 
    is executed remotely, and results are returned as a dictionary mapping frame 
    indices to segmentation mask results.

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
    with modal.enable_output():
        with app.run():
            video_segments=do_some_magic.remote(points, frame_idx, foldername, multi_resolution, is_first, is_final)
    return video_segments