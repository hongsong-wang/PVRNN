from torch.utils.data import Dataset
import numpy as np
import copy
import random

class generate_train_data(Dataset):
    def __init__(self, data_set, source_seq_len, target_seq_len, sample_start=16):
        self._data_set = data_set
        self._index_lst = list(data_set.keys())

        self._source_seq_len = source_seq_len
        self._target_seq_len = target_seq_len
        self._total_frames = self._source_seq_len + self._target_seq_len
        self._sample_start = sample_start
        self._action2index = self.get_dic_action2index()

    def __len__(self):
        return len(self._index_lst)

    def get_dic_action2index(self):
        actions = sorted(list(set([item[1] for item in self._data_set.keys()])) )
        return dict(zip(actions, range(0, len(actions))))

    def __getitem__(self, index):
        data = self._data_set[self._index_lst[index]]
        action = self._action2index.get(self._index_lst[index][1])

        # Sample somewherein the middle
        if data.shape[0] - self._total_frames <= 0:
            idx = 0
        else:
            idx = np.random.randint(self._sample_start, data.shape[0] - self._total_frames)
        # Select the data around the sampled points
        data_sel = copy.deepcopy(data[idx:idx + self._total_frames, :] )

        # decoder_inputs = data_sel[self._source_seq_len - 1:self._source_seq_len + self._target_seq_len - 1]
        # decoder_outputs = data_sel[self._source_seq_len:]
        # mirror for data augmentation
        # if random.random() > 0.5:
        #     data_sel = np.flip(data_sel, axis=0).copy()

        encoder_inputs = data_sel[0:self._source_seq_len - 1]
        decoder_target = data_sel[self._source_seq_len-1:]

        return encoder_inputs, decoder_target, action


def find_indices_srnn(data, action, batch_size):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """
    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState( SEED )

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
    T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
    prefix, suffix = 50, 100

    idx = []
    for k in range(int(batch_size/2)):
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))

    return idx

def get_batch_srnn(data, action, source_seq_len, target_seq_len, input_size):
    """
    Get a random batch of data from the specified bucket, prepare for step.
    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """
    frames = {}
    batch_size = 8 # we always evaluate 8 seeds
    frames[ action ] = find_indices_srnn( data, action, batch_size)

    subject = 5 # we always evaluate on subject 5

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    encoder_inputs = np.zeros( (batch_size, source_seq_len-1, input_size), dtype=float )
    decoder_target = np.zeros( (batch_size, target_seq_len+1, input_size), dtype=float )

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in range( batch_size ):
      _, subsequence, idx = seeds[i]
      idx = idx + 50
      data_sel = data[ (subject, action, subsequence, 'even') ]
      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

      encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
      decoder_target[i, 0:len(data_sel)-source_seq_len+1, :]  = data_sel[source_seq_len-1:, :]
    return encoder_inputs, decoder_target


def get_batch_srnn_cmu(data, action, source_seq_len, target_seq_len, input_size):
    # the previous method sample 8 seeds
    # Todo, as the result is not stable, I enlarge the batch_size of testing
    batch_size = 80
    encoder_inputs = np.zeros((batch_size, source_seq_len - 1, input_size), dtype=float)
    decoder_target = np.zeros((batch_size, target_seq_len + 1, input_size), dtype=float)
    data_sel = copy.deepcopy(data[('', action, 1, 'even')])
    total_frames = source_seq_len + target_seq_len

    SEED = 1234567890
    rng = np.random.RandomState(SEED)
    for i in xrange(batch_size):
        n, _ = data_sel.shape
        if n - total_frames <= 0:
            idx = 0
        else:
            idx = rng.randint(0, n - total_frames)

        data_sel2 = data_sel[idx:(idx + total_frames), :]
        encoder_inputs[i, :, :] = data_sel2[0:source_seq_len - 1, :]
        decoder_target[i, :, :] = data_sel2[source_seq_len - 1:, :]
    return encoder_inputs, decoder_target

if __name__ == '__main__':
    from human_36m import human_36m_dataset
    data_dir = '/home/hust/data/Human_3.6M/h3.6m/dataset'
    db = human_36m_dataset(data_dir)

    data_set = db.load_data(db.get_test_subject_ids())

    source_seq_len = 30
    target_seq_len = 20
    gen = generate_train_data(data_set, source_seq_len, target_seq_len)

    for data in gen:
        print(data[0].shape)
        data_std = np.std(data[0], axis=0)
        print(np.where(data_std < 1e-4) )

