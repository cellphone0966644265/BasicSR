# File gốc # inference_edvr.py
# Phiên bản tinh gọn tối đa phần import model.
# Giả định script này được đặt và chạy từ thư mục gốc của dự án BasicSR
# (ví dụ: /content/BasicSR/) và lớp model là 'EDVR' trong 'basicsr.archs.edvr_arch'.

import argparse
import torch
import yaml  # pip install PyYAML
import os
from pathlib import Path
import cv2  # pip install opencv-python
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

# =======================================================================================
# >>> CẤU HÌNH LOGGER <<<
# =======================================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('inference_edvr.log') # Bỏ comment để ghi log ra file
    ]
)
logger = logging.getLogger(__name__)

# =======================================================================================
# >>> IMPORT LỚP MODEL EDVR <<<
# =======================================================================================
# Giả định tên lớp thực tế là 'EDVR'. Nếu khác, hãy thay đổi tương ứng.
try:
    from basicsr.archs.edvr_arch import EDVR
    logger.info("INFO: Đã import thành công model EDVR từ basicsr.archs.edvr_arch.")
except ImportError as e:
    logger.error(f"LỖI: Không thể import EDVR từ basicsr.archs.edvr_arch. Lỗi: {e}")
    logger.error("Hãy đảm bảo bạn đã cài đặt thư viện BasicSR và cấu trúc thư mục là chính xác.")
    exit(1)
# =======================================================================================

# Xác định đường dẫn gốc của BasicSR dựa trên vị trí của script này
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASICSR_ROOT_PATH = SCRIPT_DIR.parent
except NameError:
    logger.warning("CẢNH BÁO: Không thể tự động xác định BASICSR_ROOT_PATH từ __file__.")
    logger.warning("           Giả định thư mục làm việc hiện tại (CWD) là thư mục gốc của BasicSR.")
    BASICSR_ROOT_PATH = Path.cwd()

logger.info(f"INFO: Đường dẫn gốc BasicSR được xác định là: {BASICSR_ROOT_PATH}")

# DANH BẠ MODEL: Đã được cập nhật CHÍNH XÁC theo danh sách bạn cung cấp.
AVAILABLE_MODELS: Dict[str, Dict[str, str]] = {
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

SUPPORTED_IMAGE_EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

def load_img_sequence(all_frame_paths: List[Path], current_window_indices: List[int], nframes_model: int) -> torch.Tensor:
    """
    Tải và tiền xử lý một chuỗi các frame ảnh.

    Args:
        all_frame_paths: Danh sách tất cả các đường dẫn đến frame LQ.
        current_window_indices: Danh sách các chỉ số frame cho cửa sổ hiện tại.
        nframes_model: Số lượng frame mà model yêu cầu.

    Returns:
        Một tensor chứa chuỗi các frame đã được tiền xử lý.
    
    Raises:
        ValueError: Nếu số lượng frame index không khớp với nframes_model.
        FileNotFoundError: Nếu không đọc được file ảnh.
    """
    if len(current_window_indices) != nframes_model:
        raise ValueError(f"Số frame index ({len(current_window_indices)}) không khớp nframes model ({nframes_model})")
    
    imgs_lq_list: List[torch.Tensor] = []
    for frame_idx in current_window_indices:
        img_path = all_frame_paths[frame_idx]
        img_lq: Optional[np.ndarray] = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_lq is None:
            raise FileNotFoundError(f"Không đọc được file ảnh: {img_path}")
        
        img_lq = img_lq.astype(np.float32) / 255.
        # Chuyển đổi BGR (OpenCV) sang RGB và HWC sang CHW
        img_lq_rgb_chw = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq_tensor = torch.from_numpy(img_lq_rgb_chw).float()
        imgs_lq_list.append(img_lq_tensor)
        
    return torch.stack(imgs_lq_list)

def main() -> None:
    """Hàm chính để chạy quá trình inference."""
    parser = argparse.ArgumentParser(description="Inference EDVR (Predefined Models)")
    parser.add_argument('--input', type=str, required=True, help='Thư mục chứa các frame LQ đầu vào')
    parser.add_argument('--output', type=str, required=True, help='Thư mục lưu các frame SR đầu ra')
    parser.add_argument('--model_name', type=str, required=True, choices=list(AVAILABLE_MODELS.keys()),
                        help=f"Model được định nghĩa sẵn. Lựa chọn: {', '.join(AVAILABLE_MODELS.keys())}")
    parser.add_argument('--device', type=str, default=None, help='Thiết bị sử dụng: "cuda" hoặc "cpu". Mặc định tự động phát hiện.')
    args = parser.parse_args()

    model_cfg = AVAILABLE_MODELS[args.model_name]
    model_weights_path = Path(model_cfg['model_path'])
    yaml_config_path = Path(model_cfg['yaml_path'])

    logger.info(f"INFO: Sử dụng YAML: {yaml_config_path}")
    logger.info(f"INFO: Sử dụng Model weights: {model_weights_path}")

    if not yaml_config_path.exists():
        logger.error(f"LỖI: File YAML cấu hình '{yaml_config_path}' không tồn tại."); return
    if not model_weights_path.exists():
        logger.error(f"LỖI: File trọng số model '{model_weights_path}' không tồn tại."); return
    
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error(f"LỖI: Thư mục input '{args.input}' không tồn tại hoặc không phải là thư mục."); return
    
    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"LỖI: Không thể tạo thư mục output '{output_dir}'. Lỗi: {e}"); return


    try:
        with open(yaml_config_path, 'r') as f:
            opt: Dict[str, Any] = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"LỖI: File YAML '{yaml_config_path}' không tìm thấy."); return
    except yaml.YAMLError as e:
        logger.error(f"LỖI: Không thể parse file YAML '{yaml_config_path}'. Lỗi: {e}"); return
    
    if args.device:
        device = torch.device(args.device)
    else:
        use_cpu_from_opt = opt.get('num_gpu', torch.cuda.device_count() if torch.cuda.is_available() else 0) == 0
        if torch.cuda.is_available() and not use_cpu_from_opt:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    logger.info(f"INFO: Sử dụng thiết bị: {device}")

    network_g_opt: Optional[Dict[str, Any]] = opt.get('network_g')
    if network_g_opt is None:
        logger.error(f"LỖI: Không tìm thấy cấu hình 'network_g' trong file YAML '{yaml_config_path}'."); return
        
    try:
        # Loại bỏ khóa 'type' khỏi network_g_opt trước khi truyền vào constructor
        # vì EDVR class không chấp nhận tham số 'type' một cách trực tiếp.
        network_g_params = {k: v for k, v in network_g_opt.items() if k != 'type'}
        model = EDVR(**network_g_params)
    except Exception as e:
        logger.error(f"LỖI khởi tạo model: {e}. Kiểm tra lại việc import class EDVR và file YAML cấu hình."); return
    
    logger.info(f"INFO: Khởi tạo model '{args.model_name}' thành công. Đang tải trọng số...")
    try:
        ckpt: Dict[str, Any] = torch.load(str(model_weights_path), map_location='cpu')
        
        if 'state_dict' in ckpt: state_dict = ckpt['state_dict']
        elif 'params_ema' in ckpt: state_dict = ckpt['params_ema'] 
        elif 'params' in ckpt: state_dict = ckpt['params']
        else: state_dict = ckpt

        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        
        strict_load = opt.get('path', {}).get('strict_load_g', True)
        model.load_state_dict(new_state_dict, strict=strict_load)
        logger.info(f"INFO: Tải trọng số từ '{model_weights_path}' thành công.")
    except FileNotFoundError:
        logger.error(f"LỖI: File trọng số '{model_weights_path}' không tìm thấy."); return
    except Exception as e:
        logger.error(f"LỖI khi tải trọng số: {e}"); return
    
    model.eval().to(device)

    all_lq_paths: List[Path] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        all_lq_paths.extend(sorted(list(input_dir.glob(f'*{ext}'))))
    all_lq_paths = sorted(list(set(all_lq_paths)), key=lambda p: p.name)


    if not all_lq_paths:
        logger.info(f"INFO: Không tìm thấy file ảnh nào ({', '.join(SUPPORTED_IMAGE_EXTENSIONS)}) trong thư mục '{args.input}'."); return

    nframes: Optional[int] = network_g_opt.get('num_frame')
    if nframes is None:
        logger.error("LỖI: 'num_frame' không được định nghĩa trong network_g của file YAML."); return
        
    center_offset: int = network_g_opt.get('center_frame_idx', nframes // 2)
    
    if not (0 <= center_offset < nframes): 
        logger.warning(f"CẢNH BÁO: 'center_frame_idx' ({center_offset}) không hợp lệ cho 'num_frame' ({nframes}). "
                       f"Sử dụng giá trị mặc định là {nframes // 2}.")
        center_offset = nframes // 2

    logger.info(f"INFO: Tổng số LQ frames tìm thấy: {len(all_lq_paths)}. "
                f"Model sẽ sử dụng {nframes} frames mỗi lần xử lý. "
                f"Frame SR đầu ra sẽ tương ứng với frame ở vị trí {center_offset} (0-indexed) trong cửa sổ {nframes} frames.")

    processed_count: int = 0
    num_target_frames: int = max(0, len(all_lq_paths) - (nframes - 1))
    if num_target_frames == 0 and len(all_lq_paths) > 0:
        logger.warning(f"CẢNH BÁO: Số lượng frame ({len(all_lq_paths)}) không đủ để xử lý với cửa sổ {nframes} frames.")


    with torch.no_grad():
        for i in range(len(all_lq_paths)): 
            start_win_idx = i - center_offset
            end_win_idx = start_win_idx + nframes 

            if start_win_idx < 0 or end_win_idx > len(all_lq_paths):
                continue

            current_indices = list(range(start_win_idx, end_win_idx))
            target_frame_path = all_lq_paths[i]

            try:
                input_seq = load_img_sequence(all_lq_paths, current_indices, nframes)
            except Exception as e:
                logger.error(f"Lỗi khi tải chuỗi ảnh cho frame đích {target_frame_path.name}: {e}"); continue
            
            input_seq = input_seq.unsqueeze(0).to(device) 
            
            try:
                output_tensor: torch.Tensor = model(input_seq)
            except Exception as e: 
                logger.error(f"Lỗi trong quá trình inference cho frame đích {target_frame_path.name}: {e}"); continue

            output_img_tensor = output_tensor.squeeze(0).cpu().clamp_(0, 1)
            
            output_img_np = output_img_tensor.numpy()
            output_img_hwc_rgb = np.transpose(output_img_np, (1, 2, 0)) 
            output_img_hwc_bgr = output_img_hwc_rgb[:, :, ::-1] 
            
            output_img_uint8 = (output_img_hwc_bgr * 255.0).round().astype(np.uint8)

            model_name_sanitized = args.model_name.replace(' ', '_').replace('/', '_')
            out_filename = f"{target_frame_path.stem}_{model_name_sanitized}{target_frame_path.suffix}"
            output_path = output_dir / out_filename
            
            try:
                cv2.imwrite(str(output_path), output_img_uint8)
            except Exception as e:
                logger.error(f"Lỗi khi lưu ảnh {output_path}: {e}"); continue
            
            processed_count += 1
            if processed_count % 10 == 0 or processed_count == num_target_frames:
                 logger.info(f"Đã xử lý {processed_count}/{num_target_frames} frames đích...")

    if processed_count > 0:
        logger.info(f"Hoàn thành! Đã xử lý {processed_count} frames. Kết quả được lưu tại: {args.output}")
    elif num_target_frames > 0 :
        logger.warning("Không có frame nào được xử lý thành công.")
    else:
        logger.info("Không có frame nào được xử lý do không đủ số lượng frame đầu vào cho cửa sổ của model.")


if __name__ == '__main__':
    main()
