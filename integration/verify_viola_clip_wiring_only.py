import sys
import os
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize

# 1. 경로 문제 해결
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'VIOLA'))

# 2. 필요한 VIOLA 모듈 import (robomimic 제외)
try:
    from viola_bc.policy import CenterNetSpatialTemporalPolicy
    # import robomimic.utils.file_utils as FileUtils # REMOVED
    # import robomimic.utils.torch_utils as TorchUtils # REMOVED
    # import robomimic.utils.tensor_utils as TensorUtils # REMOVED
    print("Successfully imported core VIOLA modules.")
except ImportError as e:
    print(f"Error importing core VIOLA modules: {e}")
    print("Please ensure that the VIOLA directory is structured correctly.")
    sys.exit(1)

def main():
    """VIOLA 모델을 CLIP 인코더로 교체하고 인스턴스화만 시도하여 통합을 검증합니다."""
    print("\n--- VICLIHD (VIOLA + CLIP) Wiring Confirmation (without robomimic) ---")
    
    # --- 1. Hydra로 VIOLA 설정 로드 및 CLIP 인코더로 교체 ---
    print("\n--- 1. Loading VIOLA config and overriding encoder to CLIP ---")
    try:
        with initialize(config_path="../VIOLA/viola_bc/configs", job_name="viola_clip_wiring_test"):
            cfg = compose(config_name="config", overrides=["experiment=stack_viola", "algo/model/encoder=clip_img_embedding"])
            
            print("Successfully loaded and modified config.")
            print(f"Model Name: {cfg.algo.model.name}")
            print(f"Encoder being used: {cfg.algo.model.encoder.network}")
            assert cfg.algo.model.encoder.network == 'CLIPImgEmbedding', "Encoder override failed!"
            print("Config override to CLIPImgEmbedding successful.")

    except Exception as e:
        print(f"\n[ERROR] Hydra FAILED to load or override configs: {e}")
        print("Please check the config paths and override syntax.")
        sys.exit(1)

    # --- 2. 더미 shape_meta 생성 (robomimic 의존성 제거) ---
    print("\n--- 2. Preparing dummy shape metadata for model initialization ---")
    # 모델 생성에 필요한 shape_meta를 더미로 생성
    # robomimic.utils.file_utils.get_shape_metadata_from_dataset 대신 직접 정의
    shape_meta = {
        "all_shapes": {
            "agentview_rgb": (3, 128, 128),
            "eye_in_hand_rgb": (3, 128, 128),
            "joint_states": (7,),
            "gripper_states": (2,),
            "gripper_history": (10,),
            "centernet_bbox_20": (20, 4),
        },
        "ac_dim": 7,
        "all_obs_keys": [
            "agentview_rgb", "eye_in_hand_rgb", "joint_states", 
            "gripper_states", "gripper_history", "centernet_bbox_20"
        ]
    }
    print("Dummy shape_meta created.")

    # --- 3. VIOLA+CLIP 모델 인스턴스화만 시도 ---
    print("\n--- 3. Attempting to instantiate VIOLA model with CLIP encoder ---")
    try:
        # CPU 모드 강제(노트북 테스트 환경)
        device = torch.device("cpu")
        
        # 수정된 설정으로 모델 생성
        model = CenterNetSpatialTemporalPolicy(cfg.algo.model, shape_meta).to(device)
        model.eval() # 평가 모드로 설정
        print("Successfully instantiated VIOLA model with CLIP encoder.")
        print("\n[SUCCESS] VIOLA and CLIP encoder wiring is confirmed. The model can be instantiated without robomimic.")

    except Exception as e:
        print(f"\n[ERROR] Failed during model instantiation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
