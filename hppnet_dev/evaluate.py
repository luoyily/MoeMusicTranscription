import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
import mir_eval
from scipy.stats import hmean
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import h5py
import pandas as pd

from torch.utils.data import Subset 

import hppnet.dataset as dataset_module
import torch 
from hppnet.constants import *
from hppnet.midi import save_midi
from hppnet.decoding import extract_notes, notes_to_frames
from hppnet.utils import summary, save_pianoroll, save_pianoroll_overlap

eps = sys.float_info.epsilon


def evaluate(data, model, device, onset_threshold=0.5, frame_threshold=0.5, save_path=None, save_metrics_only=False, clip_len=10240):
    metrics = defaultdict(list)

    for label in data:

        label['audio'] = label['audio'].to(device) # use [0] to unbach
        label['onset'] = label['onset'].to(device)
        label['offset'] = label['offset'].to(device)
        label['frame'] = label['frame'].to(device)
        label['velocity'] = label['velocity'].to(device)

        label['path'] = str(label['path'])

        frame_num = label['onset'].shape[-2]

        if(not save_path is None):
            os.makedirs(save_path, exist_ok=True)
            pred_path = label_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.h5')
        # load previous pred
        if(not save_path is None and os.path.exists(pred_path)):
            pred = {'onset':None, 'offset':None, 'frame':None, 'velocity': None}
            losses = {'loss/onset': None, 'loss/offset': None, 'loss/frame': None, 'loss/velocity':None}
            with h5py.File(pred_path, 'r') as h5:
                for key in pred:
                    pred[key] = torch.tensor(h5[key][:]).to(device)
                for key in losses:
                    if(key in h5):
                        losses[key] = torch.tensor(h5[key][()]).to(device)
                    else:
                        losses[key] = 0
        # get new pred
        else:
            n_step =  label['onset'].shape[-2]
            
            # 
            if(len(label['audio'].size()) > 1 or n_step <= clip_len):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    pred, losses = model.run_on_batch(label)
            # clip audio to fixed length to prevent out of memory.
            else: # when test on long audio
                print('n_step > clip_len %d '%clip_len, label['audio'].shape, label['onset'].shape)
                clip_list = [clip_len] * (n_step // clip_len)
                res = n_step % clip_len
                if(n_step > clip_len and res != 0):
                    clip_list[-1] -= (clip_len - res)//2
                    clip_list += [res + (clip_len - res)//2]

                print('clip list:', clip_list)

                begin = 0
                pred = {}
                losses = {}
                for clip in clip_list:
                    end = begin + clip
                    label_i = {}
                    label_i['audio'] = label['audio'][HOP_LENGTH*begin:HOP_LENGTH*end]
                    label_i['onset'] = label['onset'][begin:end]
                    label_i['offset'] = label['offset'][begin:end]
                    label_i['frame'] = label['frame'][begin:end]
                    label_i['velocity'] = label['velocity'][begin:end]
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        pred_i, losses_i = model.run_on_batch(label_i)

                    for key, item in pred_i.items():
                        if(key in pred):
                            pred[key] = torch.cat([pred[key], item], dim=0)
                        else:
                            pred[key] = item

                    for key, loss in losses_i.items():
                        if(key in losses):
                            losses[key] += loss * clip / n_step
                        else:
                            losses[key] = loss * clip / n_step
                    begin += clip
            # save pred
            if(not save_path is None):
                with h5py.File(pred_path, 'w') as h5:
                    for key, item in pred.items():
                        h5[key] = item.cpu().numpy()
                    for key, item in losses.items():
                        h5[key] = item.cpu().numpy()

        for key, loss in losses.items():
            metrics[key].append(loss)

        for key, value in pred.items():
            value.squeeze_(0).relu_()


        # #############################
        # # metrics of onset
        # onsets_pred = (pred['onset'] > onset_threshold).cpu().to(torch.float)
        # onsets_pred_pad = onsets_pred.clone()
        # onsets_pred_pad[:-1] += onsets_pred[1:]
        # onsets_pred_pad[1:] += onsets_pred[:-1]
        # onsets_pred_pad[:-2] += onsets_pred[2:]
        # onsets_pred_pad[2:] += onsets_pred[:-2]
        # onsets_pred_pad[:-3] += onsets_pred[3:]
        # onsets_pred_pad[3:] += onsets_pred[:-3]
        # onsets_pred_pad = torch.clip(onsets_pred_pad, 0, 1)
        # onset_pred_diff = torch.cat([onsets_pred[:1, :], onsets_pred[1:, :] - onsets_pred[:-1, :]], dim=0) == 1
        # onset_pred_diff = onset_pred_diff.to(torch.float)
        # onsets_ref = label['onset'].cpu().to(torch.float)
        # onset_ref_diff = torch.cat([onsets_ref[:1, :], onsets_ref[1:, :] - onsets_ref[:-1, :]], dim=0) == 1
        # onset_ref_diff = onset_ref_diff.to(torch.float)
        # onset_recall = torch.sum(onsets_pred_pad*onset_ref_diff) / torch.sum(onset_ref_diff)
        # onset_precision = torch.sum(onsets_pred_pad*onset_ref_diff) / torch.sum(onset_pred_diff)
        # onset_f1 = 2*onset_recall*onset_precision/(onset_recall+onset_precision)
        # metrics['metric/onsets/recall'].append( onset_recall )
        # metrics['metric/onsets/precision'].append( onset_precision )
        # metrics['metric/onsets/f1'].append( onset_f1 )
        # #############################3
        


        # pitch, interval, velocity
        p_ref, i_ref, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])
        p_est, i_est, v_est = extract_notes(pred['onset'], pred['frame'], pred['velocity'], onset_threshold, frame_threshold)

        note_ref = p_ref
                
        
        
        # time, frequency
        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)
        t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])


        ############################################
        # # find what ref notes are not matched.
        # matched_list = mir_eval.transcription.match_note_onsets(i_ref, i_est)
        # matched_ref_list = [m[0] for m in matched_list]
        # with open('not_matched_note.txt', 'a') as f:
        #     # f.write('total notes:%d\n'%(len(p_est)))
        #     for i in range(len(p_ref)):
        #         if not i in matched_ref_list:
        #             f.write('%d,%.3f,%.3f\n'%(note_ref[i], i_ref[i][1]-i_ref[i][0], v_ref[i]))
        #########################################3

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        note_recall = r
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                  offset_ratio=None, velocity_tolerance=0.1)
        metrics['metric/note-with-velocity/precision'].append(p)
        metrics['metric/note-with-velocity/recall'].append(r)
        metrics['metric/note-with-velocity/f1'].append(f)
        metrics['metric/note-with-velocity/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
        metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

        # if save_path is not:
        # metrics_path = os.path.join(save_path,'metrics_of_each_audio.txt')
        # with open(metrics_path, 'a') as f:
        #     f.write('note_recall:%.4f %s\n'%(note_recall, os.path.basename(label['path'])))

        if save_path is not None and save_metrics_only==False:
            os.makedirs(save_path, exist_ok=True)
            label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
            save_pianoroll(label_path, label['onset'], label['frame'], onset_threshold, frame_threshold, zoom=1)
            pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['onset'], pred['frame'], onset_threshold, frame_threshold, zoom=1)
            midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
            save_midi(midi_path, p_est, i_est, v_est)

            frame_overlap_path = pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.overlap.note_recall%.4f.png'%note_recall)
            save_pianoroll_overlap(frame_overlap_path, label['frame'], pred['frame'], frame_threshold, zoom=1)

            onset_overlap_path = pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.onset_overlap.note_recall%.4f.png'%note_recall)
            save_pianoroll_overlap(onset_overlap_path, label['onset'], pred['onset'], onset_threshold, zoom=1)

            pred_onset_path = pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.onset.png')
            plt.imsave(pred_onset_path, pred['onset'].cpu().numpy())

    return metrics


def evaluate_file(model_file, dataset, dataset_group, sequence_length, save_path,onset_threshold, frame_threshold, device, clip_len=10240, mini_dataset=False):

    if(save_path == None):
        group_str = dataset_group if dataset_group is not None else 'default'
        save_path = os.path.join(model_file[:-3] + "_evaluate", dataset, group_str)

    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length} # , 'device': device
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    if(mini_dataset):
        dataset = Subset(dataset, list(range(10)))

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    metrics = evaluate(tqdm(dataset), model, device, onset_threshold, frame_threshold, save_path, save_metrics_only=False, clip_len=clip_len)

    

    res = '\n' + model_file +   '\n' + datetime.now().strftime('%y%m%d-%H%M%S') + '\n\nMetrics:\n'
    res += 'evaluate dataset and group:' + str(dataset) + ', ' + str(dataset_group) + '\n'
    res += 'audio piece num: %d\n'%(len(dataset))
    res += 'sequence_len: %s, clip_len: %d\n'%(str(sequence_length), clip_len)
    res += 'onset and frame threshold: %f, %f'%(onset_threshold, frame_threshold) + '\n'


    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            res += '\n' + f'{category:>32} {name:25}: {np.mean(values):.4f} ± {np.std(values):.4f}'
            print(res)

    if(save_path != None):
        result_path = os.path.join(save_path, 'metrics_result.txt')
        with open(result_path, 'a') as f:
            f.write(res)

        # save metrics to csv
        column_dict = {}
        for key, values in metrics.items():
            # metric/note-with-offsets-and-velocity/f1
            if(key.find('loss') >= 0):
                continue
            new_key = key
            replace = {'metric/':'', '-with':'', '-and':'', 'onsets': 'on', 'offsets':'off', 'velocity':'vel'}
            for k,v in replace.items():
                new_key = new_key.replace(k, v)
            column_dict[new_key] = values
        column_dict['path'] = [os.path.split(str(data['path']))[-1] for data in dataset]
        df = pd.DataFrame.from_dict(column_dict)
        csv_path = os.path.join(save_path, 'metrics_result.csv')
        print('save to :', csv_path)
        df.to_csv(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset', nargs='?', default='MAPS')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.4, type=float)
    parser.add_argument('--frame-threshold', default=0.4, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mini-dataset', action='store_true')

    with torch.no_grad():
        evaluate_file(**vars(parser.parse_args()))
