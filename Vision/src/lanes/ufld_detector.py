import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import scipy.special
import time
from pathlib import Path

from config.utils.path_manager import path_manager # Import the consolidated path manager

# --- ARCHITECTURE ---
class ParsingNet(nn.Module):
    def __init__(self, size=(288, 800), pretrained=False, backbone='18', cls_dim=(101, 56, 4), use_aux=False):
        super(ParsingNet, self).__init__()
        self.size = size
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        self.model = models.resnet18(pretrained=pretrained)
        
        self.layer1 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1)
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        # Classification Head
        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )
        self.pool = nn.Conv2d(512, 8, 1)

    def forward(self, x):
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        fea = self.pool(x5).view(-1, 1800)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        if self.use_aux:
            return group_cls, None
        return group_cls

# --- DETECTOR WITH AGGRESSIVE LOADING ---
class UFLDDetector:
    def __init__(self, model_path=None, device='cuda'): # model_path optional
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # PATH FIX
        if model_path is None:
            # If not provided, try to find it automatically or raise an error
            # Here we assume app.py ALWAYS passes it, but for safety:
            model_path = path_manager.get_model("ufld") # Use path_manager
            
        print(f"⚡ Loading UFLD from {model_path}...")        
        
        self.input_width = 800
        self.input_height = 288
        
        # 1. Load Checkpoint for inspection
        path = Path(model_path)
        if not path.exists():
            path_alt = Path("models") / path.name
            path = path_alt if path_alt.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"❌ Cannot find {path}")

        checkpoint = torch.load(path, map_location=self.device)
        
        # Unpack
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            else: state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 2. DETECT GEOMETRY (Searching for the largest matrix)
        output_neurons = 22624 # Default
        max_params = 0
        
        # Search for the largest weight matrix (which is usually the final layer)
        for k, v in state_dict.items():
            if len(v.shape) == 2:
                params = v.shape[0] * v.shape[1]
                if params > max_params and v.shape[1] == 2048: # The final layer always receives 2048
                    max_params = params
                    output_neurons = v.shape[0]

        print(f"🔍 Detected output neurons: {output_neurons}")
        
        if output_neurons == 22624:
            print("✅ Mode: TuSimple")
            self.cls_dim = (101, 56, 4)
            self.gridding_num = 100
            self.cls_num_per_lane = 56
        elif output_neurons == 14472:
            print("✅ Mode: CULane")
            self.cls_dim = (201, 18, 4)
            self.gridding_num = 200
            self.cls_num_per_lane = 18
        else:
            print(f"⚠️ Non-standard geometry ({output_neurons}). Adjusting...")
            # Inverse calculation assuming 4 lanes and 56 rows
            grids = int(output_neurons / (56 * 4))
            self.cls_dim = (grids, 56, 4)
            self.gridding_num = grids - 1
            self.cls_num_per_lane = 56

        # 3. Instantiate Model
        self.model = ParsingNet(pretrained=False, cls_dim=self.cls_dim, use_aux=False)
        self.model.to(self.device)
        
        # 4. WEIGHT INJECTION (FORCE FEED)
        model_state = self.model.state_dict()
        new_state_dict = {}
        
        # Critical shapes that we NEED to find
        target_shapes = {
            'cls.0.weight': (2048, 1800),
            'cls.0.bias': (2048,),
            'cls.2.weight': (output_neurons, 2048),
            'cls.2.bias': (output_neurons,),
            'pool.weight': (8, 512, 1, 1),
            'pool.bias': (8,)
        }
        
        assigned_targets = set()
        matched_backbone = 0
        
        print("💉 Starting weight injection...")
        
        for k, v in state_dict.items():
            # A. Normal loading attempt by name (for Backbone)
            clean_k = k[7:] if k.startswith('module.') else k
            if clean_k in model_state and v.shape == model_state[clean_k].shape:
                new_state_dict[clean_k] = v
                matched_backbone += 1
                continue
                
            # B. Loading by SHAPE (For the missing Head)
            for target_name, target_shape in target_shapes.items():
                if target_name in assigned_targets: continue
                
                if v.shape == target_shape:
                    print(f"   -> MATCH! Assigning '{k}' to '{target_name}'")
                    new_state_dict[target_name] = v
                    assigned_targets.add(target_name)
                    break # Assigned, move to the next weight from pth

        print(f"📊 Backbone loaded: {matched_backbone} layers.")
        print(f"📊 Head loaded: {len(assigned_targets)}/6 critical components.")
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        # Generate Anchors
        if self.cls_num_per_lane == 56: 
            self.row_anchor = np.array([64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160,
                            164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208,
                            212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256,
                            260, 264, 268, 272, 276, 280, 284])
        else: 
             self.row_anchor = np.linspace(121, 288, 18).astype(int)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def detect(self, img_bgr, **kwargs):
        t_start = time.time()
        h_orig, w_orig, _ = img_bgr.shape

        # ROI Crop (Top 40%)
        crop_h = int(h_orig * 0.40) 
        img_roi = img_bgr[crop_h:, :, :]
        
        img = cv2.resize(img_roi, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor_img = tensor_img.unsqueeze(0).to(self.device)
        tensor_img = (tensor_img - self.mean) / self.std

        with torch.no_grad():
            output = self.model(tensor_img)

        t_end = time.time()
        latency = (t_end - t_start) * 1000

        lanes_points = self._process_output(output.cpu().numpy(), w_orig, h_orig, crop_offset=crop_h)

        result = img_bgr.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] 

        for i, points in enumerate(lanes_points):
            if len(points) > 2:
                for k in range(len(points) - 1):
                    cv2.line(result, points[k], points[k+1], colors[i], 3)
                    
        return result, latency

    def _process_output(self, output, w_orig, h_orig, crop_offset=0):
        pred = output[0]
        prob = scipy.special.softmax(pred[:-1, :, :], axis=0)
        idx = np.arange(self.gridding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        output_cls = np.argmax(pred, axis=0)
        loc[output_cls == self.gridding_num] = 0
        
        lanes_points = []
        roi_h = h_orig - crop_offset
        
        for lane_idx in range(4):
            points = []
            for row_idx in range(self.cls_num_per_lane):
                val = loc[row_idx, lane_idx]
                if val > 0:
                    x_net = (val * self.input_width) / (self.gridding_num - 1)
                    x_real = int(x_net * w_orig / self.input_width)
                    y_net = self.row_anchor[row_idx]
                    y_roi = int(y_net * roi_h / self.input_height)
                    y_real = y_roi + crop_offset
                    points.append((x_real, y_real))
            lanes_points.append(points)
        return lanes_points