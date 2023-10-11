import os
from datetime import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.cuda
import torch.utils.data
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred import Experiment
import numpy as np

from hppnet.dataset import PianoRollAudioDatasetH5
from hppnet.transcriber import HPPNet
from hppnet.utils import cycle
from evaluate import evaluate


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ex = Experiment('train_transcriber')


@ex.config
def data_config():
    train_data_folders = ['data_example/train']
    val_data_folders = ['data_example/val']
    test_data_folders = ['data_example/val']


@ex.config
def train_config():
    # log and resume
    logdir = 'runs/transcriber'
    resume_iteration = None

    device = 'cuda'
    iterations = 600*1000
    batch_size = 2

    # intervals
    checkpoint_interval = 2000
    validation_interval = 400
    test_interval = None

    sequence_length = 327680

    validation_length = sequence_length

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    clip_gradient_norm = 3

    # [1.0, 0.3, 0.1] preportion used for training in training set.
    training_size = 1.0
    model_name = "HPPNet"
    ex.observers.append(FileStorageObserver.create(logdir))

    # for dev test
    # sequence_length = 81920
    # validation_length = sequence_length
    # checkpoint_interval = 20
    # validation_interval = 10


@ex.config
def hpp_sp():
    model_size = 128
    SUBNETS_TO_TRAIN = ['onset_subnet', 'frame_subnet']
    onset_subnet_heads = ['onset']
    frame_subnet_heads = ['frame', 'offset', 'velocity']


@ex.named_config
def hpp_base():
    model_size = 128
    SUBNETS_TO_TRAIN = ['onset_subnet']
    onset_subnet_heads = ['onset', 'frame', 'offset', 'velocity']
    frame_subnet_heads = []


@ex.named_config
def hpp_tiny():
    model_size = 64
    SUBNETS_TO_TRAIN = ['onset_subnet']
    onset_subnet_heads = ['onset', 'frame', 'offset', 'velocity']
    frame_subnet_heads = []


@ex.named_config
def hpp_ultra_tiny():
    model_size = 48
    SUBNETS_TO_TRAIN = ['onset_subnet']
    onset_subnet_heads = ['onset', 'frame', 'offset', 'velocity']
    frame_subnet_heads = []


@ex.config
def loss_config():
    positive_weight = 2


@ex.config
def train_without_test():
    test_interval = None
    test_onset_threshold = 0.4
    test_frame_threshold = 0.4


@ex.named_config
def train_with_test():
    validation_interval = 20
    test_onset_threshold = 0.4
    test_frame_threshold = 0.4


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, batch_size, sequence_length,
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate,
          clip_gradient_norm, validation_length, validation_interval,
          test_interval, test_onset_threshold, test_frame_threshold,
          training_size, model_name, train_data_folders, val_data_folders, test_data_folders):
    print_config(ex.current_run)

    config = ex.current_run.config

    SUBNETS_TO_TRAIN = config['SUBNETS_TO_TRAIN']

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    dataset = PianoRollAudioDatasetH5(folders=train_data_folders, sequence_length=sequence_length)
    validation_dataset = PianoRollAudioDatasetH5(folders=val_data_folders, sequence_length=sequence_length)
    test_dataset = PianoRollAudioDatasetH5(folders=test_data_folders, sequence_length=sequence_length)

    train_idx = [int(x/training_size)for x in range(int(len(dataset)*training_size))]

    ex.info['training_idx'] = train_idx
    dataset = torch.utils.data.Subset(dataset, train_idx)
    ex.info['train_num'] = len(dataset)

    # Warn: Using multiple workers on Windows will lead to duplicate code execution (e.g. module import code) 
    # (here will lead to duplicate creation of CQT in the transcriber, slowing down training)
    loader = DataLoader(dataset, batch_size, shuffle=True,drop_last=True, num_workers=0)

    optimizers = {}
    if resume_iteration is None:
        model = HPPNet(config).to(device)

        for subnet in SUBNETS_TO_TRAIN:
            optimizers[subnet] = torch.optim.Adam(
                model.subnets[subnet].parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        for subnet in SUBNETS_TO_TRAIN:
            optimizers[subnet] = torch.optim.Adam(
                model.subnets[subnet].parameters(), learning_rate)
            optimizers[subnet].load_state_dict(torch.load(
                os.path.join(logdir, f'last-optimizer-state-{subnet}.pt')))

    schedulers = {}
    for subnet in SUBNETS_TO_TRAIN:
        schedulers[subnet] = StepLR(
            optimizers[subnet], step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    loop.set_description(config['model_name'])
    tqdm_dict = {}
    for i, batch in zip(loop, cycle(loader)):

        batch['audio'] = batch['audio'].to(device)
        batch['onset'] = batch['onset'].to(device)
        batch['offset'] = batch['offset'].to(device)
        batch['frame'] = batch['frame'].to(device)
        batch['velocity'] = batch['velocity'].to(device)

        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())

        for subnet in SUBNETS_TO_TRAIN:
            loss_subnet = losses[f'loss/{subnet}']
            optimizers[subnet].zero_grad()
            loss_subnet.backward()
            optimizers[subnet].step()
            schedulers[subnet].step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if (i % 10 == 0):
            tqdm_dict['train/loss'] = loss.cpu().detach().numpy()
            loop.set_postfix(tqdm_dict)

        if (i in [100, 1000, 2000, 4000, 8000] or i % 10000 == 0):
            frame_img_pred = torch.swapdims(predictions['frame'], 1, 2)
            frame_img_pred = torch.unsqueeze(frame_img_pred, dim=1)
            # => [F x T]
            frame_img_pred = torchvision.utils.make_grid(
                frame_img_pred, pad_value=0.5)
            # writer.add_image('train/step_%d_pred'%i, frame_img_pred)

            frame_img_ref = torch.swapdims(batch['frame'], 1, 2)
            frame_img_ref = torch.unsqueeze(frame_img_ref, dim=1)
            frame_img_ref = torchvision.utils.make_grid(
                frame_img_ref, pad_value=0.5)
            # writer.add_image('train/step_%d_ref'%i, frame_img_ref)

            frame_img = torch.cat([frame_img_ref[0], frame_img_pred[0]], dim=0)
            dir_path = os.path.join(logdir, 'piano_roll')
            os.makedirs(dir_path, exist_ok=True)
            plt.imsave(dir_path + '/train_step_%d.png' %
                       (i), frame_img.detach().cpu().numpy())

        ##################################
        # Validate
        if i % validation_interval == 0:
            print("validating...")
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate(validation_dataset, model, device)
                for key, value in val_metrics.items():
                    mean_val = torch.tensor(value).cpu().numpy().mean()
                    writer.add_scalar(
                        'validation/' + key.replace(' ', '_'), mean_val, global_step=i)
                    ex.log_scalar('validation/' +
                                  key.replace(' ', '_'), mean_val, i)
                # tqdm_dict['on_loss'] = '%.4f'%np.mean(val_metrics['loss/onset'])
                tqdm_dict['f_f1'] = '%.3f' % np.mean(
                    val_metrics['metric/frame/f1'])
                tqdm_dict['n_f1'] = '%.3f' % np.mean(
                    val_metrics['metric/note/f1'])
                loop.set_postfix(tqdm_dict)
            model.train()

        ##################################
        # Test
        if not test_interval is None:
            if i % test_interval == 0:
                print("testing...")
                model.eval()
                clip_len = 10240
                test_result = {}
                test_result['step'] = i
                test_result['time'] = datetime.now().strftime('%y%m%d-%H%M%S')
                test_result['dataset'] = str(test_dataset)
                test_result['dataset_group'] = test_dataset.groups
                test_result['dataset_len'] = len(test_dataset)
                test_result['clip_len'] = clip_len
                test_result['onset_threshold'] = test_onset_threshold
                test_result['frame_threshold'] = test_frame_threshold
                with torch.no_grad():
                    eval_result = evaluate(test_dataset, model, device,
                                           onset_threshold=test_onset_threshold, frame_threshold=test_frame_threshold,
                                           clip_len=clip_len,
                                           save_path=config['logdir'] +
                                           f'/model-{i}-test'
                                           )
                    for key, values in eval_result.items():
                        mean_val = np.mean(values)
                        # std_val = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                        label = 'test/' + key.replace(' ', '_')
                        writer.add_scalar(label, mean_val, global_step=i)
                        ex.log_scalar(label, mean_val, i)
                        test_result[label] = "%.2f" % (mean_val*100)
                ex.info[f'test_step_{i}'] = test_result
                model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))

            # torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            for subnet in SUBNETS_TO_TRAIN:
                torch.save(optimizers[subnet].state_dict(), os.path.join(
                    logdir, f'last-optimizer-state-{subnet}.pt'))
