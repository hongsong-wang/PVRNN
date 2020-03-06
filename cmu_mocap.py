import os
import numpy as np

def readCSVasFloat(filename):
  lines = open(filename).readlines()
  returnArray = [map(float, line.strip().split(',')) for line in lines]
  returnArray = np.array(returnArray)
  return returnArray

class cmu_mocap_dataset(object):
    def __init__(self, data_dir):
        self._actions = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking", "washwindow" ]
        self._data_dir = data_dir

    def get_train_subject_ids(self):
        return os.path.join(self._data_dir, 'train')

    def get_test_subject_ids(self):
        return os.path.join(self._data_dir, 'test')

    def get_test_actions(self):
        return zip(self._actions, range(0, len(self._actions)))

    def load_data(self, data_path):
        data_set = {}
        for action in self._actions:
            fold_path = os.path.join(data_path, action)
            file_lst = os.listdir(fold_path)
            for file in file_lst:
                subj = ''
                subact = int(file[-5])
                filename = os.path.join(fold_path, file)
                action_sequence = readCSVasFloat(filename)
                # Discard the first joint, which represents a corrupted translation
                # print(action, action_sequence.shape)
                # action_sequence = action_sequence[:, 3:]
                even_list = range(0, action_sequence.shape[0], 2)
                data_set[(subj, action, subact, 'even')] = action_sequence[even_list]
                # print((subj, action, subact, 'even'), action_sequence[even_list].shape )
                even_list = range(1, action_sequence.shape[0], 2)
                data_set[(subj, action, subact, 'odd')] = action_sequence[even_list]
        return data_set

if __name__ == '__main__':
    dataset = cmu_mocap_dataset('/home/hust/data/Human_3.6M/cmu_mocap/')
    dataset.load_data(dataset.get_test_subject_ids())