import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import modal

cuda_version = "12.2.2"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "opencv-python"
    ).run_commands(
        "python --version",
        "apt-get update && apt install -y clang",
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

def downsample_frame(frame, resolution):
    # frame: a numpy array (H,W,...) and resolution: target width and height
    return cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)

@app.function(gpu="L4", image=image, volumes={"/root/temp":vol}, timeout=3000, mounts=[])
def do_some_magic(slices, points, frame_idx, foldername, axis, multires=False, init_resolution=64, max_resolution=512):
    import subprocess

    # output = subprocess.check_output(["nvidia-smi"], text=True)
    # print(output)
    
    # cwd=os.getcwd()
    # print(cwd)
    # print(os.listdir(cwd))
    
    os.chdir(os.path.expanduser("temp/SAM_2_Medical_3D"))
    # cwd=os.getcwd()
    # print(os.listdir(cwd))

    subprocess.call(['gcc', '--version'])
    subprocess.call(['which', 'gcc'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', "-e", "."])
    subprocess.call(['gcc', '--version'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', ".[demo]"])
    subprocess.call(['gcc', '--version'])

    # TODO: get line above to work. find a way of running notebook code (with it's import statements)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    # cwd=os.getcwd()
    # print("\n"*30)
    # print(os.listdir(cwd))
    # print("\n"*30)

    # print(sys.version_info)

    # #file_dir = os.path.dirname(__file__)
    # #sys.path.append(file_dir)
    # #sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    current_directory = os.getcwd()
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + current_directory

    if current_directory not in sys.path:
        sys.path.append(current_directory)

    # #sys.path.append(os.path.expanduser("./sam2"))

    # print("*****")
    # print(sys.path)
    # print("*****")

    from sam2.build_sam import build_sam2_video_predictor
    sam2_checkpoint = "./sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    # print("*********************")
    # print(os.getcwd())

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    video_dir = f"./frames/{foldername}"

    # print(os.listdir(video_dir))
    # print("*********************")

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the src slice the user interacted with
    plt.figure(figsize=(12, 8))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))




    inference_state = predictor.init_state(video_path=video_dir)




    predictor.reset_state(inference_state)


    video_segments = {}  # video_segments contains the per-frame segmentation results

    prompts = {}  # hold all the clicks we add for visualization

    ann_frame_idx = frame_idx  # the frame index we interact with TODO: set = frame_idx (parameter)

    if not multires:
        for k,v in points.items():
            ann_obj_id = k
            coords = [[x[0], x[1]] for x in v]
            labels = [x[2] for x in v]
            coords = np.array(coords, dtype=np.float32)
            labels = np.array(labels, np.int32)
            prompts[ann_obj_id] = coords, labels
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=coords,
                labels=labels,
            )

            # show the results on the current (interacted) frame
            plt.figure(figsize=(12, 8))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            show_points(coords, labels, plt.gca())
            for i, out_obj_id in enumerate(out_obj_ids):    
                show_points(*prompts[out_obj_id], plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    else:
        print("IN CONTAINER: Multiresolution segmentation")
        if axis == 0:
            original_image = slices[frame_idx, :, :]
        elif axis == 1:
            original_image = slices[:, frame_idx, :]
        elif axis == 2:
            original_image = slices[:, :, frame_idx]
        else:
            raise ValueError("Invalid view axis. Must be 0,1, or 2.")
            return

        original_h, original_w = original_image.shape
        current_res = init_resolution
        if original_h < current_res or original_w < current_res:
            raise ValueError("Initial resolution is too high for the input image.")
            return
        
        if len(points) == 0:
            raise ValueError("No points provided for multiresolution segmentation.")
            return
        
        color_mapping = {
            "red": 1,
            "blue": 2,
            "green": 3,
            "orange": 4,
            "purple": 5,
            "cyan": 6,
            "magenta": 7,
            "yellow": 8,
            "black": 9,
            "gray": 10
        }

        seeds = {}
        print(points)
        for ann_obj_id, pts in points.items():
            for (x, y, color, *_) in pts:
                color = str(color)
            obj_id = color_mapping.get(color.lower(), 1)
            temp_scale_x = int(x * current_res / original_w)
            temp_scale_y = int(y * current_res / original_h)
            seeds.setdefault(obj_id, []).append((temp_scale_x, temp_scale_y, 1))
        
        def downsample_volume(res):
            if axis == 0:
                N = slices.shape[0]
                vol = np.zeros((N, res, res), dtype=original_image.dtype)
                for i in range(N):
                    vol[i] = cv2.resize(slices[i,:,:], (res,res), interpolation=cv2.INTER_AREA)
                return vol
            elif axis == 1:
                H = slices.shape[1]
                vol = np.zeros((H, res, res), dtype=slices.dtype)
                for j in range(H):
                    vol[j] = cv2.resize(slices[:,j,:], (res,res), interpolation=cv2.INTER_AREA)
                return vol
            elif axis == 2:
                W = slices.shape[2]
                vol = np.zeros((W, res, res), dtype=slices.dtype)
                for k in range(W):
                    vol[k] = cv2.resize(slices[:,:,k], (res, res), interpolation=cv2.INTER_AREA)
                return vol
        
        # inner function that performs one iteration
        def iteration(resolution, seeds):
            if resolution > max(original_h, original_w):
                raise ValueError("Resolution is too high for the input image.")
                return

            downsampled_volume = downsample_volume(resolution)
            points = seeds

            result_container = {}
            def completion_callback(video_segments):
                result_container['video_segments'] = video_segments

            # do segmentation here
            for k, v in seeds.items():
                ann_obj_id = k
                points = np.array([[x[0], x[1]] for x in v], dtype=np.float32)
                labels = np.array([x[2] for x in v], dtype=np.int32)
                # modify pts/labels based on the resolution
                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=0):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            def next_iteration():
                # Extract new seeds from the downsampled mask.
                new_seeds = {}
                for obj_id in seeds.keys():
                    seg_mask_down = video_segments[frame_idx].get(obj_id)
                    if seg_mask_down is not None:
                        if seg_mask_down.ndim == 3 and seg_mask_down.shape[0] == 1:
                            seg_mask_down = seg_mask_down[0]
                        mask_uint8 = (seg_mask_down > 0).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            contour = max(contours, key=cv2.contourArea)
                            # use centroid of the largest segmentation mask
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                new_seeds[obj_id] = [(cx, cy, 1)]
                            else:
                                new_seeds[obj_id] = seeds[obj_id]
                        else:
                            # if no contours found then use the original seed
                            new_seeds[obj_id] = seeds[obj_id]
                    else:
                        # if segmentation mask is not returned then use the original seed
                        new_seeds[obj_id] = seeds[obj_id]
                
                for obj_id, pts in new_seeds.items():
                    mpl_color = {1:"r",2:"b",3:"g",4:"orange",5:"purple",6:"c",7:"m",8:"y",9:"k",10:"gray"}.get(obj_id, "r")
                    xs = [int(p[0]*(original_w/resolution)) for p in pts]
                    ys = [int(p[1]*(original_h/resolution)) for p in pts]
                
                new_res = resolution * 2
                if new_res > max(original_h, original_w):
                    return
                else:
                    iteration(new_res, new_seeds)

            next_iteration()
        
        iteration(init_resolution, seeds)
            

                
                
                    

        # while current_res <= max_resolution:
        #     down_frame = downsample_frame(orig_frame, current_res)
        #     scaled_points = {}
        #     for k, v in points.items():
        #         scaled = []
        #         for (x, y, flag) in v:
        #             sx = int(x * current_res / original_w)
        #             sy = int(y * current_res / original_h)
        #             scaled.append((sx, sy, flag))
        #         scaled_points[k] = scaled
            
        #     for k, v in scaled_points.items():    
        #         ann_obj_id = k
        #         pts = np.array([[x[0], x[1]] for x in v], dtype=np.float32)
        #         labels = np.array([x[2] for x in v], dtype=np.int32)
        #         # modify pts/labels based on the resolution
        #         _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        #             inference_state=inference_state,
        #             frame_idx=ann_frame_idx,
        #             obj_id=ann_obj_id,
        #             points=pts,
        #             labels=labels,
        #         )
        #         # You might accumulate or update prompts/results here if desired.
        #     # Optionally, visualize the current iteration.
        #     plt.figure(figsize=(12, 8))
        #     plt.title(f"Multires iteration at resolution {current_res} on frame {ann_frame_idx}")
        #     plt.imshow(down_frame)
        #     current_res *= 2
    
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
    
    return video_segments

def segment(slices, points, frame_idx, foldername, axis_str_suffix, multires=False):
    """
    segmentation functionality.
    """
    if axis_str_suffix.lower() == "axial":
        axis = 0
    elif axis_str_suffix.lower() == "coronal":
        axis = 1
    elif axis_str_suffix.lower() == "sagittal":
        axis = 2
    else:
        raise ValueError("Invalid view axis. Must be 'axial', 'coronal', or 'sagittal'.")
        return

    with modal.enable_output():
        with app.run():
            video_segments=do_some_magic.remote(slices, points, frame_idx, foldername, axis, multires)
    # video_dir = f"./{foldername}"
    # vis_frame_stride = 15
    # frame_names = [
    #     p for p in os.listdir(video_dir)
    #     if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    # ]
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    # plt.close("all")
    # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    #     plt.figure(figsize=(6, 4))
    #     plt.title(f"frame {out_frame_idx}")
    #     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    # plt.show()
    # print("finished in handler, returning to controller")
    return video_segments