from __future__ import print_function
import sys
import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from human_36m import human_36m_dataset
from cmu_mocap import cmu_mocap_dataset
from batch_sample import generate_train_data, get_batch_srnn, get_batch_srnn_cmu
from seq2seq import EncodeDecodeModel
import torch_utils
from utils import Logger, AverageMeter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu only')
    parser.add_argument('--dataset', default='h36m', help='h36m or cmu')
    parser.add_argument('--data_dir', default='/home/hust/data/Human_3.6M/h3.6m/dataset')
    parser.add_argument('--data_dir_cmu', default='/home/hust/data/Human_3.6M/cmu_mocap/')
    parser.add_argument('--num_joint_cmu', type=int, default=38, help='number of joints for cmu dataset')
    parser.add_argument('--log_file', default='log_train.txt')
    parser.add_argument('--save_path', default='/temp/wanghongsong/data/savetemp/motpred')

    parser.add_argument('--source_seq_len', type=int, default=50, help='length of encode sequence')
    parser.add_argument('--target_seq_len', type=int, default=25, help='length of output decode sequence')
    parser.add_argument('--num_joint', type=int, default=32, help='input size at each timestep')
    parser.add_argument('--hid_size', type=int, default=1024, help='hidden size of RNN')
    parser.add_argument('--rnn_unit', default='gru', help='gru or lstm')
    parser.add_argument('--num_layer', type=int, default=1, help='number of rnn layers')
    parser.add_argument('--residual', action='store_true', default=True, help='residual rnn for decoder')
    parser.add_argument('--veloc', action='store_true', default=True, help='model velocity input')
    parser.add_argument('--loss_type', type=int, default=0, help='0:expmap, 1:quaternion')
    parser.add_argument('--std_mask', action='store_true', default=True, help='set joint dimension with std < 1e-4 to zero')
    parser.add_argument('--out_dropout', type=float, default=0.1, help='dropout ratio for output prediction')
    parser.add_argument('--pos_embed', action='store_true', default=True, help='use position embedding')
    parser.add_argument('--pos_embed_dim', type=int, default=96, help='dimension of position_dim, default 96, if <=0, use one-hot')

    parser.add_argument('--batch_size', type=int, default=128, help='make sure batch size large than 8 due to testing, only 180 sequences')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate reduce ratio')
    parser.add_argument('--num_max_epoch', type=int, default=50000, help='number of epoch for all training samples')
    parser.add_argument('--step_size', type=int, default=30000, help='step epoch to reduce learning rate')

    args = parser.parse_args()
    args.log_dir = 'log/' + args.dataset
    sys.stdout = Logger(os.path.join(args.log_dir, args.log_file))
    print(args)

    if args.dataset == 'h36m':
        dataset = human_36m_dataset
    if args.dataset == 'cmu':
        dataset = cmu_mocap_dataset
        args.data_dir = args.data_dir_cmu
        args.num_joint = args.num_joint_cmu

    input_size = 3*args.num_joint
    model = EncodeDecodeModel(input_size, args.hid_size, args.num_layer, args.rnn_unit, args.out_dropout, args.std_mask,
                              args.learning_rate, args.step_size, args.gamma, residual=args.residual, cuda=(not args.cpu),
                              veloc=args.veloc, loss_type=args.loss_type, pos_embed=args.pos_embed, pos_embed_dim=args.pos_embed_dim)

    dataset = dataset(args.data_dir)
    train_set = dataset.load_data(dataset.get_train_subject_ids())
    test_set = dataset.load_data(dataset.get_test_subject_ids())

    train_gen = generate_train_data(train_set, args.source_seq_len, args.target_seq_len, sample_start=0 if args.dataset == 'cmu' else 16)

    total_err_pre = 1e3
    for epoch_i in range(int(args.num_max_epoch)):
        model.scheduler_step()
        losses = AverageMeter()
        for source, target, action in DataLoader(train_gen, batch_size=args.batch_size, shuffle=True):
            if not args.cpu:
                source = source.cuda()
                target = target.cuda()

            # Discard the first joint, which represents a corrupted translation
            source = source[:,:,3:]
            target = target[:,:,3:]
            # convert seq to (time, batch, dim)
            source = source.permute(1, 0, 2).float()
            target = target.permute(1, 0, 2).float()

            loss = model.train(source, target)
            losses.update(loss, source.size(1))

        print('train epoch %d, loss = %0.5f, lr = %0.5f' % (epoch_i + 1, losses.val, model.model_optimizer.param_groups[-1]['lr']))

        # === Validation with srnn's seeds ===
        if epoch_i % 16 ==0 and epoch_i > 0:
            total_err_lst = []
            print("{0: <18} |".format("milliseconds"), end="")
            for ms in [80, 160, 320, 400, 560, 1000]:
                print(" {0:5d} |".format(ms), end="")
            print()
            for action, action_idx in dataset.get_test_actions():
                # Evaluate the model on the test batches
                if args.dataset == 'h36m':
                    source_tst, target_tst = get_batch_srnn(test_set, action, args.source_seq_len, args.target_seq_len, 3*args.num_joint+3)
                else:
                    source_tst, target_tst = get_batch_srnn_cmu(test_set, action, args.source_seq_len, args.target_seq_len, 3 * args.num_joint+3)

                source_tst = torch.tensor(source_tst).to(source.device)
                target_tst = torch.tensor(target_tst).to(source.device)

                # Discard the first joint, which represents a corrupted translation
                source_tst = source_tst[:, :, 3:]
                target_tst = target_tst[:, :, 3:]
                source_tst = source_tst.permute(1, 0, 2).float()
                target_tst = target_tst.permute(1, 0, 2).float()

                pred_target = model.eval(source_tst, target_tst)

                target_tst = torch_utils.tensor_expmap_to_euler(target_tst)

                # Convert from exponential map to Euler angles
                pred_target = torch_utils.tensor_expmap_to_euler(pred_target)

                # global rotation the first 3 entries are also not considered in the error
                error = torch.pow(target_tst[1:, :, 3:] - pred_target[:,:, 3:], 2)
                error = torch.sqrt(torch.sum(error, dim=-1) )
                error = torch.mean(error, dim=1)
                mean_mean_errors = error.cpu().detach().numpy()
                total_err_lst.append(np.mean(mean_mean_errors) )

                # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                print("{0: <18} |".format(action), end="")
                for ms in [1, 3, 7, 9, 13, 24]:
                    if mean_mean_errors.shape[0] >= ms + 1:
                        print(" {0:.3f} |".format(mean_mean_errors[ms]), end="")
                    else:
                        print("   n/a |", end="")
                print() # start new line

            total_err = np.mean(total_err_lst)
            if epoch_i > 5000: #  and total_err < total_err_pre
                save_file = '%s_%s_epoch_%d_err_%0.5f.pt' % (args.save_path, args.dataset, epoch_i + 1, total_err)
                # torch.save(model.model.state_dict(), save_file)
                total_err_pre = total_err

if __name__ == '__main__':
    main()