import numpy as np
import torch

def get_note_duration(frames):
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
    assert(bins == 88)
    durs = torch.zeros(frames.shape)
    durs[:,-1] = frames[:,-1]
    for i in range(T-1):
        durs[:, -(i+2)] = (durs[:, -(i+1)] + 1) * frames[:, -(i+2)]

    return durs.to(torch.int)


def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
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
    mid = (onsets[1:-1] >= onsets[2:]).float() * (onsets[1:-1] >= onsets[:-2]).float()
    onsets = torch.cat([left, mid, right], dim=0).float() * onsets

    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    # => [T x 88]
    durs = get_note_duration(frames.T).T

    pitches = []
    intervals = []
    velocities = []

    T = onsets.shape[0]

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        if(onset + 1 >= T):
            offset = onset
        else:
            offset = onset + min(durs[onset+1, pitch], 1000)
        offset = min(offset, T)

        velocity_samples = []
        onset_end = onset
        while onsets[onset_end, pitch].item():
            velocity_samples.append(velocity[onset_end, pitch].item())
            onset_end += 1
            if onset_end >= T:
                break

        # while onsets[offset, pitch].item() or frames[offset, pitch].item():
        #     if onsets[offset, pitch].item():
        #         velocity_samples.append(velocity[offset, pitch].item())
        #     offset += 1
        #     if offset == onsets.shape[0]:
        #         break
        #     if(offset - frame > 1000): # ignore more than 1000 frames. ()
        #         break

        # consider all pred onset has a note.
        # if offset > onset:
        pitches.append(pitch)
        intervals.append([onset, max(onset+1, offset)])
        velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs
