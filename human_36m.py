import numpy as np

def readCSVasFloat(filename):
  lines = open(filename).readlines()
  # returnArray = [map(float, line.strip().split(',')) for line in lines] # python2.7
  returnArray = [list(map(float, line.strip().split(','))) for line in lines]
  returnArray = np.array(returnArray)
  return returnArray

class human_36m_dataset(object):
    def __init__(self, data_dir):
        self._actions = ["walking", "eating", "smoking", "discussion",  "directions",
                      "greeting", "phoning", "posing", "purchases", "sitting",
                      "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]
        self._data_dir = data_dir
        self._action2index = self.get_dic_action2index()

    def get_dic_action2index(self):
        return dict(zip(self._actions, range(0, len(self._actions)) ))

    def get_train_subject_ids(self):
        return [1, 6, 7, 8, 9, 11]

    def get_test_subject_ids(self):
        return [5]

    def load_data(self, subjects):
        data_set = {}
        for subj in subjects:
            for action in self._actions:
                for subact in [1, 2]:  # subactions
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self._data_dir, subj, action, subact)
                    action_sequence = readCSVasFloat(filename)
                    # take 1/2 of that, for a final rate of 25fps
                    # https://github.com/una-dinosauria/human-motion-prediction/issues/8
                    even_list = range(0, action_sequence.shape[0], 2)
                    data_set[(subj, action, subact, 'even')] = action_sequence[even_list]
                    even_list = range(1, action_sequence.shape[0], 2)
                    data_set[(subj, action, subact, 'odd')] = action_sequence[even_list]
        return data_set

    def get_test_actions(self):
        return zip(self._actions, range(0, len(self._actions)))

if __name__ == '__main__':
    data_dir = '/home/hust/data/Human_3.6M/h3.6m/dataset'
    db = human_36m_dataset(data_dir)

    db.load_data(db.get_test_subject_ids() )