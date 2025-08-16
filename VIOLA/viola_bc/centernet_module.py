from detectron2.config import get_cfg
import sys, os
from pathlib import Path
# Ensure third_party absolute paths based on repo root
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO / 'third_party' / 'Detic'))
sys.path.insert(0, str(_REPO / 'third_party' / 'Detic' / 'third_party' / 'CenterNet2' / 'projects' / 'CenterNet2'))
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from viola_bc.detic_configs import detic_cfg, BUILDIN_METADATA_PATH, BUILDIN_CLASSIFIER
from viola_bc.detic_models import DefaultPredictor, VisualizationDemo
from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.engine import DefaultPredictor
from detic.modeling.utils import reset_cls_test
from detectron2.utils.visualizer import Visualizer

from detectron2.structures import Boxes, Instances

def load_centernet_rpn(nms=0.5):
    centernet_cfg = get_cfg()
    add_centernet_config(centernet_cfg)
    add_detic_config(centernet_cfg)
    centernet_cfg.merge_from_file(str(_REPO / 'third_party' / 'Detic' / 'configs' / 'Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml'))
    centernet_cfg.MODEL.WEIGHTS = str(_REPO / 'third_party' / 'Detic' / 'models' / 'Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth')
    centernet_cfg.MODEL.META_ARCHITECTURE = 'ProposalNetwork'
    centernet_cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST = 512
    # centernet_cfg.MODEL.CENTERNET.INFERENCE_TH = 0.1
    centernet_cfg.MODEL.CENTERNET.NMS_TH_TEST = nms
    # cfg.MODEL.DEVICE='cpu'
    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    centernet_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # be more permissive
    centernet_cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    centernet_cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    # cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3]
    # cfg.MODEL.RPN.IOU_THRESHOLDS = [0.7, 1.0]
    centernet_cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(centernet_cfg)
    
    metadata = MetadataCatalog.get("proposal")
    metadata.thing_classes = [''] # Change here to try your own vocabularies!

    return predictor
