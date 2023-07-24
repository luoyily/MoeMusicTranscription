import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import TuneIcon from '@mui/icons-material/Tune';
import SaveAltIcon from '@mui/icons-material/SaveAlt';
import Autocomplete from '@mui/material/Autocomplete';
import { Button, TextField } from '@mui/material'
import LoadingButton from '@mui/lab/LoadingButton';
import Alert from '@mui/material/Alert';
import Radio from '@mui/material/Radio';
import DirectionsRunIcon from '@mui/icons-material/DirectionsRun';
import DownloadIcon from '@mui/icons-material/Download';
import * as React from 'react';
import { useRef, useEffect, useState } from 'react';

import Slider from '@mui/material/Slider';
import MuiInput from '@mui/material/Input';

// const backendUrl = 'http://127.0.0.1:8000/';
const backendUrl = '';

function InferAlert(props) {
    //props.inferState:finish|backendError|none|missingParam
    switch (props.inferState) {
        case 'finish':
            return <Alert severity="success">Your inference task has been completed, click to download the result!</Alert>
        case 'backendError':
            return <Alert severity="error">Backend Error</Alert>
        case 'missingParam':
            return <Alert severity="info">Missing required parameters</Alert>
        default:
            return null
    }
}
function ModelSelector(props) {
    // props.setStateAction: React.SetStateAction
    const [modelOptions, setModelOptions] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        fetch(`${backendUrl}hppnet_models`)
            .then(response => response.json())
            .then(data => {
                setModelOptions(data['models']);
                console.log(data['models']);
                setIsLoading(false);
            })
            .catch(error => {
                console.error('Error:', error);
                setIsLoading(false);
            });
    }, []);

    if (isLoading) {
        return <Typography>Loading...</Typography>;
    }
    // props.setStateAction(modelOptions[0])
    return (
        <Autocomplete
            size="small"
            id="combo-box-demo"
            options={modelOptions}
            sx={{ flexGrow: 1 }}
            renderInput={(params) => <TextField variant="standard" {...params} />}
            onChange={(event, newValue) => {
                props.setStateAction(newValue);
            }}
        />
    );
}
function FileUploader(props) {
    const fileInputRef = useRef(null);
    function handleButtonClick() {
        fileInputRef.current.click();
    }
    function handleFileChange(event) {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            fetch(`${backendUrl}uploadfile`, {
                method: 'POST',
                body: formData
            })
                .then(response => {
                    if (response.ok) {
                        props.setStateAction(true)
                    }
                })
                .catch(error => {
                    console.error(error);
                });
        }
    }

    return (
        <Box
            sx={{ display: 'flex' }}>
            <label htmlFor="fileInput" style={{ display: 'flex' }}>
                <Button
                    sx={{ flexGrow: 1 }}
                    variant='outlined' color='info' onClick={handleButtonClick}>Upload </Button>
            </label>
            <input
                type="file"
                id="fileInput"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleFileChange}
            />
        </Box>

    );
}

const PianoTrans = () => {
    // const modelOptions = ['model-142000-mestro-maps_mus-my-f0.912n0.973.pt', 'model-192000-f0.919 n0.974-piano_only.pt'];
    const [filePath, setFilePath] = useState(null);
    const [modelName, setModelName] = useState('');
    const [deviceSelectedValue, setDeviceSelectedValue] = useState('gpu');
    const [onsetValue, setOnsetValue] = useState(0.5);
    const [frameValue, setFrameValue] = useState(0.5);
    const [gpuID, setgpuID] = useState(0);

    const [isFileUpload, setIsFileUpload] = useState(false);
    const [inferState, setInferState] = useState(null);
    const [infering, setInfering] = useState(false)
    const [isInferNotFin, setIsInferNotFin] = useState(true)
    const inferResultLinkRef = useRef(null)

    const handleInferResultDownload = () => {
        inferResultLinkRef.current.click()
    }
    const handleDeviceChange = (event) => {
        // console.log(event.target.value)
        setDeviceSelectedValue(event.target.value);
    };
    const handleOnsetSliderChange = (event, newValue) => {
        setOnsetValue(newValue);
    };

    const handleOnsetInputChange = (event) => {
        setOnsetValue(event.target.value === '' ? '' : Number(event.target.value));
    };

    const handleFrameSliderChange = (event, newValue) => {
        setFrameValue(newValue);
    };

    const handleFrameInputChange = (event) => {
        setFrameValue(event.target.value === '' ? '' : Number(event.target.value));
    };
    const handleGPUIDChange = (event) => {
        setgpuID(event.target.value === '' ? '' : Number(event.target.value));
    };
    const handlefilePathChange = (event) => {
        setFilePath(event.target.value);

        // console.log(event.target.value)
        // console.log(filePath)
    }
    const handleOnsetBlur = () => {
        if (onsetValue < 0) {
            setOnsetValue(0);
        } else if (onsetValue > 1) {
            setOnsetValue(1);
        }
    };
    const handleFrameBlur = () => {
        if (frameValue < 0) {
            setFrameValue(0);
        } else if (frameValue > 1) {
            setFrameValue(1);
        }
    };
    const handleInferClick = () => {
        if (filePath || isFileUpload) {
            setInferState(null)
            setInfering(true)
        } else {
            setInferState('missingParam')
        }
        console.log(inferState)
        let inferTask = {
            "file_path": filePath,
            "model_name": modelName,
            "device": deviceSelectedValue,
            "onset_t": onsetValue,
            "frame_t": frameValue,
            "gpu_id": gpuID
        }
        console.log(inferTask)
        fetch(`${backendUrl}infer_hppnet`, {
            method: 'POST',
            responseType: 'blob',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inferTask)
        })
            .then(response => {
                if (response.ok) {
                    console.log('infer success')
                    return response.blob()
                    
                } else { 
                    setInferState('backendError') 
                    console.log('be')
                    return null
                };
            })
            .then((blob) => {
                if (blob){
                    const url = window.URL.createObjectURL(new Blob([blob]),);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'result.mid';
                    inferResultLinkRef.current = link
                    console.log(link)
                    console.log(inferResultLinkRef)
                    setInfering(false)
                    setIsInferNotFin(false)
                    setInferState('finish')
                }
                
            })
            .catch(error => {
                console.error('Error:', error);
            });

    }
    return (
        <Box
            sx={{
                flexGrow: 1,
                backgroundColor: '#FFF',
                padding: 2,
                borderRadius: 4,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                maxHeight: '95%'
            }}> <Box
                sx={{ flexGrow: 1 }}>
                <Typography
                    sx={{ display: 'flex', marginBottom: 2 }}>
                    <UploadFileIcon></UploadFileIcon>File Input:
                </Typography>
                <Typography sx={{ marginLeft: 3, marginBottom: 2 }}>
                    Enter the local file path or upload a file.
                </Typography>
                <Box
                    sx={{ display: 'flex', flexDirection: 'row', maxWidth: '83%' }}
                >
                    <TextField
                        size="small"
                        id="outlined-basic" label="Filepath" variant="outlined"
                        sx={{ flexGrow: 1, marginInline: 3 }}
                        onChange={handlefilePathChange}
                    />
                    <FileUploader setStateAction={setIsFileUpload}></FileUploader>
                </Box>
            </Box>


            <Box
                sx={{ flexGrow: 1, marginTop: 1, maxWidth: '85%' }}>
                <Typography
                    sx={{ display: 'flex' }}>
                    <TuneIcon />Model Config:
                </Typography>
                <Box
                    sx={{ margin: 3 }}>
                    <Box
                        sx={{ display: 'flex', flexDirection: 'row' }}>
                        <Typography>Model:</Typography>
                        <ModelSelector setStateAction={setModelName}></ModelSelector>
                    </Box>
                    <Box
                        sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography>
                            Device:
                        </Typography>
                        <Radio size='small'
                            checked={deviceSelectedValue === 'cpu'}
                            onChange={handleDeviceChange}
                            value="cpu" />
                        <Typography>CPU</Typography>
                        <Radio size='small'
                            checked={deviceSelectedValue === 'gpu'}
                            onChange={handleDeviceChange}
                            value="gpu" />
                        <Typography>GPU</Typography>
                        <Typography
                            sx={{ marginLeft: 2 }}
                        >
                            GPU ID
                        </Typography>
                        <MuiInput
                            sx={{ marginLeft: 2 }}
                            value={gpuID}
                            size="small"
                            onChange={handleGPUIDChange}
                            inputProps={{
                                step: 1,
                                min: 0,
                                max: 100,
                                type: 'number',
                                'aria-labelledby': 'input-slider',
                            }}
                        />
                    </Box>
                    <Box
                        sx={{ display: 'flex', flexDirection: 'column' }}>
                        <Box
                            sx={{ display: 'flex' }}>
                            <Typography>
                                OnsetThreshold:
                            </Typography>
                            <Slider
                                size='small'
                                min={0}
                                max={1}
                                step={0.01}
                                value={typeof onsetValue === 'number' ? onsetValue : 0.5}
                                onChange={handleOnsetSliderChange}
                                aria-labelledby="input-slider"
                                sx={{ marginInline: 2 }}
                            />
                            <MuiInput
                                value={onsetValue}
                                size="small"
                                onChange={handleOnsetInputChange}
                                onBlur={handleOnsetBlur}
                                inputProps={{
                                    step: 0.1,
                                    min: 0,
                                    max: 1,
                                    type: 'number',
                                    'aria-labelledby': 'input-slider',
                                }}
                            />
                        </Box>
                        <Box
                            sx={{ display: 'flex' }}>
                            <Typography>
                                FrameThreshold:
                            </Typography>
                            <Slider
                                size='small'
                                min={0}
                                max={1}
                                step={0.01}
                                value={typeof frameValue === 'number' ? frameValue : 0.5}
                                onChange={handleFrameSliderChange}
                                aria-labelledby="input-slider"
                                sx={{ marginInline: 2 }}
                            />
                            <MuiInput
                                value={frameValue}
                                size="small"
                                onChange={handleFrameInputChange}
                                onBlur={handleFrameBlur}
                                inputProps={{
                                    step: 0.1,
                                    min: 0,
                                    max: 1,
                                    type: 'number',
                                    'aria-labelledby': 'input-slider',
                                }}
                            />
                        </Box>
                    </Box>

                </Box>

            </Box>


            <Box
                sx={{ flexGrow: 1 }}
            >
                <Typography
                    sx={{ display: 'flex' }}>
                    <SaveAltIcon />Infer&Download:
                </Typography>

                <Box sx={{ margin: 2 }}>
                    <Box sx={{ margin: 2 }}><InferAlert inferState={inferState}></InferAlert></Box>
                    <LoadingButton sx={{ marginLeft: 2 }} variant='outlined' color='info'
                        onClick={handleInferClick}
                        loading={infering}
                        startIcon={<DirectionsRunIcon />}
                        loadingPosition="start"
                    >
                        Run Infer</LoadingButton>
                    <Button sx={{ marginLeft: 2 }} variant='outlined' color='success'
                        disabled={isInferNotFin}
                        startIcon={<DownloadIcon />}
                        onClick={handleInferResultDownload}
                    >
                        Download Result</Button>
                </Box>

            </Box>

        </Box>);
}

export default PianoTrans;