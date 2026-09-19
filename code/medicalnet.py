from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
        dtype=out.dtype, device=out.device,
    )
    return torch.cat([out, zero_pads], dim=1)


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation,
                      stride=stride, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride,
            dilation=dilation, padding=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu(out)


class MedicalNetBackbone(nn.Module):
    def __init__(self, block, layers, shortcut_type="B"):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.out_channels = 512 * block.expansion  # 2048 for resnet50

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block, planes=planes * block.expansion, stride=stride
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


_ARCH_REGISTRY = {
    10: (BasicBlock, [1, 1, 1, 1], "B"),
    18: (BasicBlock, [2, 2, 2, 2], "A"),
    34: (BasicBlock, [3, 4, 6, 3], "A"),
    50: (Bottleneck, [3, 4, 6, 3], "B"),
}


def build_backbone(depth: int, shortcut_type: str = None):
    if depth not in _ARCH_REGISTRY:
        raise ValueError(
            f"No confirmed pretrained checkpoint for depth={depth} in the "
            f"23-dataset MedicalNet release; only {sorted(_ARCH_REGISTRY)} are "
            f"available there. (101/152/200 exist as architectures but verify "
            f"their shortcut_type against the repo before using them.)"
        )
    block, layers, default_shortcut = _ARCH_REGISTRY[depth]
    return MedicalNetBackbone(block, layers, shortcut_type=shortcut_type or default_shortcut)


def resnet18_backbone(shortcut_type="A"):
    return MedicalNetBackbone(BasicBlock, [2, 2, 2, 2], shortcut_type=shortcut_type)


def resnet50_backbone(shortcut_type="B"):
    return MedicalNetBackbone(Bottleneck, [3, 4, 6, 3], shortcut_type=shortcut_type)


class MedicalNetClassifier(nn.Module):
    def __init__(self, backbone: MedicalNetBackbone, num_classes=1, dropout=0.2, hidden_dims=None):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool3d(1)

        dims = [backbone.out_channels] + list(hidden_dims or [])
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def load_pretrained_backbone(backbone: MedicalNetBackbone, ckpt_path: str, device="cpu", verbose=True):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        name = k[len("module."):] if k.startswith("module.") else k
        if name.startswith("conv_seg"):
            continue
        cleaned[name] = v

    result = backbone.load_state_dict(cleaned, strict=False)
    if verbose:
        print(f"[MedicalNet] loaded {len(cleaned)} tensors from {ckpt_path}")
        print(f"[MedicalNet] missing_keys:    {result.missing_keys}")
        print(f"[MedicalNet] unexpected_keys: {result.unexpected_keys}")
        if result.missing_keys or result.unexpected_keys:
            print("[MedicalNet] WARNING: non-empty missing/unexpected keys -- "
                  "double check the checkpoint depth/shortcut_type match what "
                  "you built (e.g. resnet_18_23dataset.pth needs depth=18, "
                  "shortcut_type='A').")
    return backbone


_LAYER_ORDER = ["conv1", "layer1", "layer2", "layer3", "layer4"]


def freeze_all(model: MedicalNetClassifier):
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_from(model: MedicalNetClassifier, start_layer="layer4"):
    freeze_all(model)
    if start_layer is not None:
        assert start_layer in _LAYER_ORDER, f"start_layer must be one of {_LAYER_ORDER} or None"
        idx = _LAYER_ORDER.index(start_layer)
        for name in _LAYER_ORDER[idx:]:
            for p in getattr(model.backbone, name).parameters():
                p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_medicalnet_classifier(ckpt_path, device="cpu", depth=18, unfreeze="layer4", dropout=0.5, hidden_dims=None):
    backbone = build_backbone(depth)
    load_pretrained_backbone(backbone, ckpt_path, device=device)
    model = MedicalNetClassifier(backbone, num_classes=1, dropout=dropout, hidden_dims=hidden_dims)
    unfreeze_from(model, start_layer=unfreeze)
    print(f"[MedicalNet] resnet{depth} | trainable params: {trainable_param_count(model):,} "
          f"/ {sum(p.numel() for p in model.parameters()):,} total")
    return model.to(device)