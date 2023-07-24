import onnxruntime
import numpy as np
import librosa
from mido import Message, MidiFile, MidiTrack
# print(onnxruntime.get_device())
# print(onnxruntime.get_available_providers())


class HPPNetOnnx:
    def __init__(self, onset_onnx_path=None, frame_onnx_path=None, model_type='sp'
                 ,providers= ['DmlExecutionProvider'],provider_options=None,
                 cqt_backend='librosa',cqt_onnx_path=None):
        """This class is mainly written for hppnet sp, 
        where the model constants are fixed in the following code.
        Except for the sp model, the inference results of the subnet need to be processed further on your own. 
        Args:
            model_type(str):sp,base,tiny,ultra_tiny
            providers:eg. ['DmlExecutionProvider']/['CPUExecutionProvider']
            provider_options: eg. [{'device_id': 0}]
            cqt_backend:'librosa','nnAudio'
        """
        self.onset_onnx_path = onset_onnx_path
        self.frame_onnx_path = frame_onnx_path
        self.clip_len = 512
        self.model_type = model_type
        self.providers = providers
        self.provider_options = provider_options
        self.cqt_backend = cqt_backend
        self.cqt_onnx_path = cqt_onnx_path

    def load_model(self):
        self.onset_sess = onnxruntime.InferenceSession(
            self.onset_onnx_path, providers=self.providers, provider_options=self.provider_options)
        if self.frame_onnx_path:
            self.frame_sess = onnxruntime.InferenceSession(
                self.frame_onnx_path, providers=self.providers, provider_options=self.provider_options)
        if self.cqt_backend == 'nnAudio':
            self.cqt_sess = onnxruntime.InferenceSession(
            self.cqt_onnx_path, providers=self.providers, provider_options=self.provider_options)

    def to_cqt(self, audio):
        """audio to cqt db

        Args:
            audio (np.ndarray [shape=(..., n)])

        Returns:
            cqt_db (np.ndarray): (1, T, 352)
        """
        if self.cqt_backend=='librosa':
            e = 2**(1/24)
            cqt = np.abs(librosa.cqt(audio, sr=16000, hop_length=320, fmin=27.5/e, n_bins=88*4, bins_per_octave=4*12,
                                    window='hann', pad_mode='reflect'))
            cqt_db = librosa.power_to_db(cqt)
            cqt_db = np.transpose(cqt_db, (0, 2, 1))
            return cqt_db
        elif self.cqt_backend=='nnAudio':
            input_name = self.cqt_sess.get_inputs()[0].name
            output_name = self.cqt_sess.get_outputs()[0].name
            output = self.cqt_sess.run([output_name], {input_name: audio})
            return output[0]
    

    def forward_cqt(self, cqt_db):
        """subnet infer

        Args:
            cqt_db (np.ndarray): (1, T, 352)


        Returns:
            dict: {'onset':[onset_subnet_output,...],'frame':[frame_subnet_output,...]}
        """
        cqt_db = np.expand_dims(cqt_db, 1)
        output_names = {
            'sp': {'onset': [self.onset_sess.get_outputs()[0].name],
                   'frame': [self.frame_sess.get_outputs()[0].name, self.frame_sess.get_outputs()[2].name]}
        }
        onset_output = self.onset_sess.run(
            output_names[self.model_type]['onset'], {'input': cqt_db})
        if self.frame_onnx_path:
            frame_output = self.frame_sess.run(
                output_names[self.model_type]['frame'], {'input': cqt_db})
        else:
            frame_output = None
        return {'onset': onset_output, 'frame': frame_output}

    def clip_audio(self, audio):
        """clip audio to list

        Args:
            audio (np.ndarray [1,n])

        Returns:
            list: [audio_slice,...]
        """
        audio_length = audio.shape[1]
        n_step = (audio_length - 1) // 320 + 1
        if n_step <= self.clip_len:
            return audio
        else:
            audio_slices = []
            clip_list = [self.clip_len] * (n_step // self.clip_len)
            res = n_step % self.clip_len
            # If not divisible, the last two slices are equally divided into two segments of the same length
            if (n_step > self.clip_len and res != 0):
                clip_list[-1] -= (self.clip_len - res)//2
                clip_list += [res + (self.clip_len - res)//2]
            begin = 0
            for clip in clip_list:
                end = begin + clip
                audio_i = audio[0][320*begin:320*end]
                audio_i = np.expand_dims(audio_i, 0)
                audio_slices.append(audio_i)
                begin += clip
            return audio_slices

    def inference_audio_file(self, auido_fn):
        """

        Args:
            auido_fn (str): auido filename

        Returns:
            tuple: (onset_cat[T*88],frame_cat,velocity_cat) 
        """
        audio, sr = librosa.load(auido_fn, sr=16000, mono=True)
        audio = np.expand_dims(audio, 0)
        slices = self.clip_audio(audio)
        outputs = []
        for a in slices:
            cqt_db = self.to_cqt(a)
            model_out = self.forward_cqt(cqt_db)
            outputs.append(model_out)
        if self.model_type == 'sp':
            onset_cat = np.concatenate([i['onset'][0]for i in outputs], axis=2)[0][0]
            frame_cat = np.concatenate([i['frame'][0]for i in outputs], axis=2)[0][0]
            velocity_cat = np.concatenate([i['frame'][1] for i in outputs], axis=2)[0][0]
        return (np.maximum(0,onset_cat), np.maximum(0,frame_cat), np.maximum(0,velocity_cat))


class HPPNetNumpyDecoder:
    """numpy implementation of the hppnet decoder for use with the onnx version,
    modified from hppnet/decoding.py;hppnet/midi.py;mir_eval
    Includes inference result processing, midi export and other functions
    """

    def __init__(self) -> None:
        pass

    def hz_to_midi(self, freqs):
        '''Convert Hz to MIDI numbers

        Parameters
        ----------
        freqs : number or ndarray
            Frequency/frequencies in Hz

        Returns
        -------
        midi : number or ndarray
            MIDI note numbers corresponding to input frequencies.
            Note that these may be fractional.
        '''
        return 12.0 * (np.log2(freqs) - np.log2(440.0)) + 69.0

    def midi_to_hz(self, midi):
        '''Convert MIDI numbers to Hz

        Parameters
        ----------
        midi : number or ndarray
            MIDI notes

        Returns
        -------
        freqs : number or ndarray
            Frequency/frequencies in Hz corresponding to `midi`
        '''
        return 440.0 * (2.0 ** ((midi - 69.0)/12.0))

    def get_note_duration(self, frames):
        # inputs: ndarray [88 x T]
        # outputs: ndarray [88 x T]
        '''
            input:
                [[0,0,1,1,1,1,0],
                [1,1,0,0,0,0,1],
                [1,0,1,0,1,1,0],
                [0,1,0,1,1,1,0],
                [1,1,0,1,0,1,1]]
            output:
                [[0 0 4 3 2 1 0]
                [2 1 0 0 0 0 1]
                [1 0 1 0 2 1 0]
                [0 1 0 3 2 1 0]
                [2 1 0 1 0 2 1]]
        '''
        bins, T = frames.shape
        assert (bins == 88)
        durs = np.zeros(frames.shape, dtype=np.int32)
        durs[:, -1] = frames[:, -1]
        for i in range(T-1):
            durs[:, -(i+2)] = (durs[:, -(i+1)] + 1) * frames[:, -(i+2)]

        return durs

    def save_midi(self, path, pitches, intervals, velocities):
        """
        Save extracted notes as a MIDI file
        Parameters
        ----------
        path: the path to save the MIDI file
        pitches: np.ndarray of bin_indices
        intervals: list of (onset_index, offset_index)
        velocities: list of velocity values
        """
        file = MidiFile()
        track = MidiTrack()
        file.tracks.append(track)
        ticks_per_second = file.ticks_per_beat * 2.0

        events = []
        for i in range(len(pitches)):
            events.append(
                dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
            events.append(
                dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
        events.sort(key=lambda row: row['time'])

        last_tick = 0
        for event in events:
            current_tick = int(event['time'] * ticks_per_second)
            velocity = int(event['velocity'] * 127)
            if velocity > 127:
                velocity = 127
            pitch = int(round(self.hz_to_midi(event['pitch'])))
            track.append(Message(
                'note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
            last_tick = current_tick

        file.save(path)

    def extract_notes(self, onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
        """
        Finds the note timings based on the onsets and frames information

        Parameters
        ----------
        onsets: np.ndarray, shape = [frames, bins]
        frames: np.ndarray, shape = [frames, bins]
        velocity: np.ndarray, shape = [frames, bins]
        onset_threshold: float
        frame_threshold: float

        Returns
        -------
        pitches: np.ndarray of bin_indices
        intervals: np.ndarray of rows containing (onset_index, offset_index)
        velocities: np.ndarray of velocity values
        """

        # only peaks are consider as onsets
        left = onsets[:1, :] >= onsets[1:2, :]
        right = onsets[-1:, :] >= onsets[-2:-1, :]
        mid = (onsets[1:-1] >= onsets[2:]) * (onsets[1:-1] >= onsets[:-2])
        onsets = np.concatenate([left, mid, right], axis=0) * onsets

        onsets = (onsets > onset_threshold).astype(np.uint8)
        frames = (frames > frame_threshold).astype(np.uint8)
        onset_diff = np.concatenate(
            [onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1

        # => [T x 88]
        durs = self.get_note_duration(frames.T).T

        pitches = []
        intervals = []
        velocities = []

        T = onsets.shape[0]
        # d = np.transpose(np.nonzero(onset_diff))
        # d2 = np.nonzero(onset_diff)
        for nonzero in np.transpose(np.nonzero(onset_diff)):
            frame = nonzero[0]
            pitch = nonzero[1]

            onset = frame
            if onset + 1 >= T:
                offset = onset
            else:
                offset = onset + min(durs[onset+1, pitch], 1000)
            offset = min(offset, T)

            velocity_samples = []
            onset_end = onset
            while onsets[onset_end, pitch]:
                velocity_samples.append(velocity[onset_end, pitch])
                onset_end += 1
                if onset_end >= T:
                    break

            # consider all pred onset has a note.

            pitches.append(pitch)
            intervals.append([onset, max(onset+1, offset)])
            velocities.append(np.mean(velocity_samples)
                              if len(velocity_samples) > 0 else 0)

        return np.array(pitches), np.array(intervals), np.array(velocities)
    
    def export_infer_result_to_midi(self,onset,frame,velocity,save_path,onset_threshold=0.5, frame_threshold=0.5):
        p_est, i_est, v_est = self.extract_notes(onset, frame, velocity, onset_threshold, frame_threshold)
        scaling = 320 / 16000
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([self.midi_to_hz(21 + midi) for midi in p_est])
        self.save_midi(save_path, p_est, i_est, v_est)

