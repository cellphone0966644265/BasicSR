# inference_edvr.py
# Phiên bản tinh gọn tối đa phần import model.
# Giả định script này được đặt và chạy từ thư mục gốc của dự án BasicSR
# (ví dụ: /content/BasicSR/) và lớp model là 'EDVR' trong 'basicsr.archs.edvr_arch'.

import argparse
import torch
import yaml # pip install PyYAML
import os
from pathlib import Path
import cv2 # pip install opencv-python
import numpy as np

# =======================================================================================
# >>> IMPORT LỚP MODEL EDVR <<<
# =======================================================================================
# Dòng import này dựa trên đường dẫn bạn cung cấp (/content/BasicSR/basicsr/archs/edvr_arch.py)
# và xác nhận tên lớp là 'EDVR'.
# Đảm bảo bạn chạy script từ thư mục gốc BasicSR (ví dụ: cd /content/BasicSR/)
# hoặc Python có thể tìm thấy gói 'basicsr'.

from basicsr.archs.edvr_arch import EDVR as YOUR_ACTUAL_EDVR_CLASS
# print("INFO: Sử dụng model EDVR từ basicsr.archs.edvr_arch.") # Bỏ comment nếu muốn xác nhận
# =======================================================================================

# Xác định đường dẫn gốc của BasicSR dựa trên vị trí của script này
try:
    SCRIPT_DIR = Path(__file__).resolve().parent 
    BASICSR_ROOT_PATH = SCRIPT_DIR.parent      
except NameError: 
    print("CẢNH BÁO: Không thể tự động xác định BASICSR_ROOT_PATH từ __file__.")
    print("           Giả định thư mục làm việc hiện tại (CWD) là thư mục gốc của BasicSR.")
    BASICSR_ROOT_PATH = Path.cwd()

# DANH BẠ MODEL: Đã được cập nhật CHÍNH XÁC theo danh sách bạn cung cấp.
AVAILABLE_MODELS = {
    "EDVR_L_DeblurComp_REDS": {
        "model_path": str(BASICSR_ROOT_PATH / "experiments/pretrained_models/EDVR/EDVR_L_deblurcomp_REDS_official-0e988e5c.pth"),
        "yaml_path": str(BASICSR_ROOT_PATH / "options/test/EDVR/test_EDVR_L_deblurcomp_REDS.yml")
    },
    "EDVR_L_Deblur_REDS": {
        "model_path": str(BASICSR_ROOT_PATH / "experiments/pretrained_models/EDVR/EDVR_L_deblur_REDS_official-ca46bd8c.pth"),
        "yaml_path": str(BASICSR_ROOT_PATH / "options/test/EDVR/test_EDVR_L_deblur_REDS.yml")
    },
    "EDVR_L_x4_SRblur_REDS": {
        "model_path": str(BASICSR_ROOT_PATH / "experiments/pretrained_models/EDVR/EDVR_L_x4_SRblur_REDS_official-983d7b8e.pth"),
        "yaml_path": str(BASICSR_ROOT_PATH / "options/test/EDVR/test_EDVR_L_x4_SRblur_REDS.yml")
    },
    "EDVR_L_x4_SR_REDS": {
        "model_path": str(BASICSR_ROOT_PATH / "experiments/pretrained_models/EDVR/EDVR_L_x4_SR_REDS_official-9f5f5039.pth"),
        "yaml_path": str(BASICSR_ROOT_PATH / "options/test/EDVR/test_EDVR_L_x4_SR_REDS.yml")
    },
    "EDVR_L_x4_SR_Vimeo90K": {
        "model_path": str(BASICSR_ROOT_PATH / "experiments/pretrained_models/EDVR/EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth"),
        "yaml_path": str(BASICSR_ROOT_PATH / "options/test/EDVR/test_EDVR_L_x4_SR_Vimeo90K.yml")
    },
    "EDVR_M_x4_SR_REDS": { 
        "model_path": str(BASICSR_ROOT_PATH / "experiments/pretrained_models/EDVR/EDVR_M_x4_SR_REDS_official-32075921.pth"),
        "yaml_path": str(BASICSR_ROOT_PATH / "options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml")
    }
}

def load_img_sequence(all_frame_paths, current_window_indices, nframes_model):
    if len(current_window_indices) != nframes_model:
        raise ValueError(f"Số frame index ({len(current_window_indices)}) không khớp nframes model ({nframes_model})")
    imgs_lq_list = []
    for frame_idx in current_window_indices:
        img_path = all_frame_paths[frame_idx]
        img_lq = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_lq is None: raise FileNotFoundError(f"Không đọc được: {img_path}")
        img_lq = img_lq.astype(np.float32) / 255.
        img_lq = torch.from_numpy(np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))).float()
        imgs_lq_list.append(img_lq)
    return torch.stack(imgs_lq_list)

def main():
    parser = argparse.ArgumentParser(description="Inference EDVR (Predefined Models)")
    parser.add_argument('--input', type=str, required=True, help='Input LQ frames folder')
    parser.add_argument('--output', type=str, required=True, help='Output SR frames folder')
    parser.add_argument('--model_name', type=str, required=True, choices=list(AVAILABLE_MODELS.keys()),
                        help=f"Predefined model. Choices: {', '.join(AVAILABLE_MODELS.keys())}")
    parser.add_argument('--device', type=str, default=None, help='Device: "cuda" or "cpu"')
    args = parser.parse_args()

    model_cfg = AVAILABLE_MODELS[args.model_name]
    model_weights_path = Path(model_cfg['model_path'])
    yaml_config_path = Path(model_cfg['yaml_path'])

    # print(f"INFO: BasicSR root path: {BASICSR_ROOT_PATH}") # Bỏ comment nếu cần debug
    # print(f"INFO: Sử dụng YAML: {yaml_config_path}")
    # print(f"INFO: Sử dụng Model weights: {model_weights_path}")

    if not yaml_config_path.exists():
        print(f"LỖI: YAML '{yaml_config_path}' không tồn tại."); return
    if not model_weights_path.exists():
        print(f"LỖI: Model weights '{model_weights_path}' không tồn tại."); return
    if not Path(args.input).is_dir():
        print(f"LỖI: Thư mục input '{args.input}' không tồn tại."); return
    Path(args.output).mkdir(parents=True, exist_ok=True)

    with open(yaml_config_path, 'r') as f: opt = yaml.safe_load(f)
    
    if args.device: device = torch.device(args.device)
    else: device = torch.device('cuda' if torch.cuda.is_available() and opt.get('num_gpu', 0) != 0 else 'cpu')
    print(f"INFO: Sử dụng device: {device}")

    network_g_opt = opt.get('network_g')
    if network_g_opt is None:
        print(f"LỖI: Không có 'network_g' trong YAML '{yaml_config_path}'."); return
        
    try:
        model = YOUR_ACTUAL_EDVR_CLASS(**network_g_opt) 
    except Exception as e:
        print(f"LỖI khởi tạo model: {e}. Kiểm tra import và YAML."); return
    
    print(f"INFO: Khởi tạo model '{args.model_name}' thành công. Đang tải trọng số...")
    try:
        ckpt = torch.load(str(model_weights_path), map_location='cpu')
        if 'state_dict' in ckpt: state_dict = ckpt['state_dict']
        elif 'params_ema' in ckpt: state_dict = ckpt['params_ema']
        elif 'params' in ckpt: state_dict = ckpt['params']
        else: state_dict = ckpt
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        strict_load = opt.get('path', {}).get('strict_load_g', True)
        model.load_state_dict(new_state_dict, strict=strict_load)
        print(f"INFO: Tải trọng số từ '{model_weights_path}' thành công.")
    except Exception as e:
        print(f"LỖI tải trọng số: {e}"); return
    
    model.eval().to(device)

    all_lq_paths = sorted(list(Path(args.input).glob('*.[jp][pn]g'))) 
    if not all_lq_paths:
        print(f"INFO: Không tìm thấy ảnh (png, jpg, jpeg) trong '{args.input}'."); return

    nframes = network_g_opt['num_frame']
    center_offset = network_g_opt.get('center_frame_idx', nframes // 2)
    if not (0 <= center_offset < nframes): 
        print(f"CẢNH BÁO: center_frame_idx không hợp lệ. Dùng giá trị mặc định {nframes // 2}.")
        center_offset = nframes // 2

    print(f"INFO: Tổng LQ frames: {len(all_lq_paths)}. Model dùng {nframes} frames/lần, SR frame ở vị trí {center_offset}.")

    processed_count = 0
    num_target_frames = max(0, len(all_lq_paths) - (nframes - 1))

    with torch.no_grad():
        for i in range(len(all_lq_paths)): 
            start_win_idx = i - center_offset
            end_win_idx = start_win_idx + nframes
            if start_win_idx < 0 or end_win_idx > len(all_lq_paths): continue

            current_indices = list(range(start_win_idx, end_win_idx))
            try:
                input_seq = load_img_sequence(all_lq_paths, current_indices, nframes)
            except Exception as e:
                print(f"Lỗi tải chuỗi cho frame đích {all_lq_paths[i].name}: {e}"); continue
            
            input_seq = input_seq.unsqueeze(0).to(device)
            try:
                output_tensor = model(input_seq)
            except Exception as e: 
                print(f"Lỗi inference cho frame đích {all_lq_paths[i].name}: {e}"); continue

            output_img = output_tensor.squeeze(0).cpu().clamp_(0, 1).numpy()
            output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0)) * 255.0
            output_img = output_img.round().astype(np.uint8)

            out_filename = f"{all_lq_paths[i].stem}_{args.model_name.replace(' ','_')}{all_lq_paths[i].suffix}"
            cv2.imwrite(str(Path(args.output) / out_filename), output_img)
            processed_count += 1
            if processed_count % 50 == 0 or processed_count == num_target_frames:
                 print(f"Đã xử lý {processed_count}/{num_target_frames} frames đích...")

    print(f"Hoàn thành! Đã xử lý {processed_count} frames. Kết quả tại: {args.output}")

if __name__ == '__main__':
    main()
