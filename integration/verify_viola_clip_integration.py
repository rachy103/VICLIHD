import sys
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from hydra import compose, initialize

# 1. 경로 문제 해결
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'VIOLA'))

# 2. 필요한 VIOLA 및 기타 모듈 import
try:
    from viola_bc.policy import CenterNetSpatialTemporalPolicy
    import robomimic.utils.torch_utils as TorchUtils
    print("Successfully imported VICLIHD modules.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that the VICLIHD, VIOLA, and hdp directories are structured correctly.")
    sys.exit(1)

def main():
    """VIOLA의 인코더를 CLIP으로 교체하고 포워드 패스를 실행하여 통합을 검증합니다."""
    print("\n--- VICLIHD (VIOLA + CLIP) Integration Verification ---")
    
    # --- 1. Hydra로 VIOLA 설정 로드 및 CLIP 인코더로 교체 ---
    print("\n--- 1. Loading VIOLA config and overriding encoder to CLIP ---")
    try:
        # VIOLA의 설정 파일 경로를 기준으로 Hydra 초기화
        with initialize(config_path="../VIOLA/viola_bc/configs", job_name="viola_clip_integration_test"):
            # 'stack_viola' 실험 설정을 기본으로 사용하고, 모델의 인코더를 'clip_img_embedding'으로 덮어쓰기
            cfg = compose(config_name="config", overrides=["experiment=stack_viola", "algo/model/encoder=clip_img_embedding"])
            
            print("Successfully loaded and modified config.")
            print(f"Model Name: {cfg.algo.model.name}")
            print(f"Encoder being used: {cfg.algo.model.encoder.network}")
            assert cfg.algo.model.encoder.network == 'CLIPImgEmbedding', "Encoder override failed!"
            print("Config override to CLIPImgEmbedding successful.")

    except Exception as e:
        print(f"\n[ERROR] Hydra FAILED to load or override configs: {e}")
        print("Please check the config paths and override syntax.")
        return

    # --- 2. 더미 데이터 및 shape_meta 생성 ---
    print("\n--- 2. Preparing dummy data and shape metadata for model initialization ---")
    # 모델 생성에 필요한 shape_meta를 더미로 생성
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
    
    # 모델 입력으로 사용할 더미 배치 데이터 생성 (Batch, Time, ...)
    B, T = 2, 10
    dummy_batch = {
        "obs": {
            "agentview_rgb": torch.randn(B, T, 3, 128, 128),
            "eye_in_hand_rgb": torch.randn(B, T, 3, 128, 128),
            "joint_states": torch.randn(B, T, 7),
            "gripper_states": torch.randn(B, T, 2),
            "gripper_history": torch.randn(B, T, 10),
            "centernet_bbox_20": torch.randn(B, T, 20, 4),
        },
        "actions": torch.randn(B, T, 7)
    }
    print("Dummy data and shape_meta created.")

    # --- 3. VIOLA+CLIP 모델 생성 및 포워드 패스 검증 ---
    print("\n--- 3. Initializing model and running forward pass ---")
    try:
        # 테스트는 노트북/CPU 모드로 강제
        device = TorchUtils.get_torch_device(try_to_use_cuda=False)
        
        # 수정된 설정으로 모델 생성
        model = CenterNetSpatialTemporalPolicy(cfg.algo.model, shape_meta).to(device)
        model.eval() # 평가 모드로 설정
        print("Successfully initialized VIOLA model with CLIP encoder.")
        
        # 더미 데이터를 모델과 동일한 디바이스로 이동
        def move_to_device(obj, device):
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                seq = [move_to_device(v, device) for v in obj]
                return type(obj)(seq) if not isinstance(obj, tuple) else tuple(seq)
            return obj
        dummy_batch = move_to_device(dummy_batch, device)
        
        # 포워드 패스 실행
        with torch.no_grad():
            output = model(dummy_batch)

        # 분포 객체 처리: 샘플을 뽑아 형태를 검증
        if hasattr(output, 'sample'):
            out_tensor = output.sample()
        else:
            out_tensor = output

        print(f"Forward pass successful. Output shape: {getattr(out_tensor, 'shape', type(out_tensor))}")
        expected_shape = (B, T, shape_meta['ac_dim'])
        assert getattr(out_tensor, 'shape', None) == expected_shape, \
            f"Output shape mismatch! Expected {expected_shape}, got {getattr(out_tensor, 'shape', type(out_tensor))}"
        print("Output shape is correct.")

    except Exception as e:
        print(f"\n[ERROR] Failed during model initialization or forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[SUCCESS] VIOLA and CLIP encoder integration is verified. The pipeline is correctly wired.")

if __name__ == "__main__":
    main()
