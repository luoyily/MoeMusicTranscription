import torch
from hppnet.constants import *
from hppnet.transcriber import HPPNet


model_path = 'ckpt/model-sena.pt'
model = torch.load(model_path).eval()

# config = {
#     'device': 'cuda', 'SUBNETS_TO_TRAIN': ['onset_subnet', 'frame_subnet'],
#     'onset_subnet_heads': ['onset'], 'frame_subnet_heads': ['frame', 'offset', 'velocity'],
#     'model_size': 128
#     }
# model_for_onnx = HPPNet(config=config)

model_for_onnx = HPPNet(config=model.config)

model_for_onnx.load_state_dict(model.state_dict())
model_for_onnx = model_for_onnx.cuda()
model_for_onnx.eval()
model_for_onnx.inference_mode = True


i = torch.randn((1, 1, 512, 352)).cuda()
input_names = ['input']
dynamic_axes = {'input': {2: 'T'}}

m_onset = model_for_onnx.subnet_onset
torch.onnx.export(m_onset, i, 'hppnet_onset_subnet.onnx',
                  input_names=input_names, dynamic_axes=dynamic_axes)

if 'frame_subnet' in model.config['SUBNETS_TO_TRAIN']:
    m_frame = model_for_onnx.subnet_frame
    torch.onnx.export(m_frame, i, 'hppnet_frame_subnet.onnx',
                      input_names=input_names, dynamic_axes=dynamic_axes)
