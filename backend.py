import os
from typing import Union
import webbrowser
from threading import Thread
import time

import uvicorn
from fastapi import FastAPI,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from pydantic import BaseModel

from hppnet.hppnet_onnx import HPPNetNumpyDecoder,HPPNetOnnx

# Configs:
port = 8612

# This is the test function used in development
cqt_backend = 'librosa' #librosa or nnAudio(need cqt onnx patch)
cqt_onnx_path = ''

class HppnetInferTask(BaseModel):
    file_path:Union[str, None] = None
    model_name:str
    device:str
    onset_t:float
    frame_t:float
    gpu_id:Union[int, None] = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="./webui/static"), name="static")
app.mount("/assets", StaticFiles(directory="./webui/assets"), name="assets")

# For node dev
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hppnet_onnx = None # type: HPPNetOnnx
hppnet_onnx_state = {}
hppnet_decoder = HPPNetNumpyDecoder()

@app.get("/")
def root():
    with open('./webui/index.html','r',encoding='utf-8') as f:
        html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)

def check_hppnet_onnx_state_change(model_name,device,gpu_id):
    new_state = {"model":model_name,"device":device,"gpu_id":gpu_id}
    return not bool(new_state == hppnet_onnx_state)

def init_hppnet_onnx(onset_onnx,frame_onnx,device,gpu_id):
    global hppnet_onnx
    if device=='gpu':
            provider_options = [{'device_id': gpu_id}] if gpu_id else None
            hppnet_onnx = HPPNetOnnx(onset_onnx,frame_onnx,provider_options=provider_options,
                                     cqt_backend=cqt_backend,cqt_onnx_path=cqt_onnx_path)
    else:
        hppnet_onnx = HPPNetOnnx(onset_onnx,frame_onnx,providers=['CPUExecutionProvider'],
                                 cqt_backend=cqt_backend,cqt_onnx_path=cqt_onnx_path)

@app.get('/hppnet_models')        
def get_available_hppnet_models():
    return {"models":os.listdir('./hppnet/models')}

@app.post('/infer_hppnet')
def run_hppnet_infer(hppnet_infer_task:HppnetInferTask):
    file_path=hppnet_infer_task.file_path if hppnet_infer_task.file_path else './backend_temp/temp.bin'
    model_name=hppnet_infer_task.model_name
    device=hppnet_infer_task.device
    onset_t=hppnet_infer_task.onset_t
    frame_t=hppnet_infer_task.frame_t
    gpu_id=hppnet_infer_task.gpu_id
    print(file_path)
    global hppnet_onnx
    onset_onnx = f'./hppnet/models/{model_name}/onset_subnet.onnx'
    frame_onnx = f'./hppnet/models/{model_name}/frame_subnet.onnx'
    output_mid = './backend_temp/temp.mid'
    # Check if hppnet_onnx is initialised
    if hppnet_onnx:
        if check_hppnet_onnx_state_change(model_name,device,gpu_id):
            del hppnet_onnx
            init_hppnet_onnx(onset_onnx,frame_onnx,device,gpu_id)
    else:
        init_hppnet_onnx(onset_onnx,frame_onnx,device,gpu_id)
    hppnet_onnx_state['model'] = model_name
    hppnet_onnx_state['device'] = device
    hppnet_onnx_state['gpu_id'] = gpu_id
    # inference
    hppnet_onnx.load_model()
    onset,frame,velocity = hppnet_onnx.inference_audio_file(file_path)
    hppnet_decoder.export_infer_result_to_midi(onset,frame,velocity,output_mid,onset_t,frame_t)
    return FileResponse(output_mid,media_type='blob')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    f = open('./backend_temp/temp.bin','wb')
    f.write(contents)
    f.close()
    return {"filename": file.filename}


def run_server(port):
    uvicorn.run(app, host="0.0.0.0", port=port)

def open_browser(port):
    time.sleep(3)
    webbrowser.open(f'http://127.0.0.1:{port}/')

if __name__ == "__main__":

    t1 = Thread(target=run_server, args=[port])
    t2 = Thread(target=open_browser, args=[port])

    t1.start()
    t2.start()

    t1.join()
    # t2.join()