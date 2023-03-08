import os
import numpy as np
from common.model_class.Feature import DYNAMIC_FEATURES

class Dataset:
    def __init__(self, arg):
        self.arg = arg

        in_full_file = os.path.join(arg.dataset_dir, 'full.npz')
        in_train_file = os.path.join(arg.dataset_dir, 'train-{}.npz'.format(self.arg.exp_id))
        in_valid_file = os.path.join(arg.dataset_dir, 'valid-{}.npz'.format(self.arg.exp_id))

        full_dataset = self._make_dataset(in_full_file)
        self.full_ids, self.full_xs, self.full_ys = full_dataset

        train_dataset = self._make_dataset(in_train_file)
        self.train_ids, self.train_xs, self.train_ys = train_dataset

        valid_dataset = self._make_dataset(in_valid_file)
        self.valid_ids, self.valid_xs, self.valid_ys = valid_dataset

        valid_sample_indices = np.random.choice(len(self.valid_ids), min(len(self.valid_ids), 1000), replace=False)
        self.valid_sample_ids = self.valid_ids[valid_sample_indices]
        self.valid_sample_xs = self.valid_xs[valid_sample_indices]
        self.valid_sample_ys = self.valid_ys[valid_sample_indices]

        proj_dir = arg.dataset_dir.split('preprocess')[0]
        DATASET_BASE = arg.dataset_dir.split('preprocess')[1].split('_P')[0].split('/')[1]
        trajectory_file = os.path.join(proj_dir, 'trajectory-data/' + DATASET_BASE + '/trajectory_data/full_norm.npz')

        if os.path.exists(trajectory_file):
            trajectory_data = self._make_dataset_trajectory(trajectory_file)
            self.trajectory_ids, self.trajectory_xs, self.trajectory_ys, self.trajectory_used_cols = trajectory_data

    def _make_dataset(self, in_file):
        dataset = np.load(in_file, allow_pickle=True)['dataset'][()]

        xs = None
        ys = np.eye(2)[dataset['LABEL']]

        cols = list(dataset.keys())
        # print(cols)

        used_cols = list()

        ids = dataset['PID']
        for col in cols:
            if col in ['PID', 'LABEL']:
                continue
            elif col == 'AGE':
                pass
            elif col.split('_')[0] in DYNAMIC_FEATURES[self.arg.feature_set]:
                pass
            else:
                continue

            used_cols.append(col)

            feature = np.array([dataset[col]]).T
            if xs is not None:
                xs = np.concatenate((xs, feature), axis=1)
            else:
                xs = feature

        # print(used_cols)

        return ids, xs, ys

    def _make_dataset_trajectory(self, in_file):
        dataset = np.load(in_file, allow_pickle=True)['dataset'][()]

        xs = None
        ys = np.eye(2)[dataset['LABEL']]

        cols = list(dataset.keys())
        #print(cols)

        used_cols = list()

        ids = dataset['PID']
        for col in cols:
            if col in ['PID', 'LABEL']:
                continue
            elif col == 'AGE':
                pass
            elif col.split('_')[0] in DYNAMIC_FEATURES[self.arg.feature_set]:
                pass
            else:
                continue
            used_cols.append(col)
            feature = np.array([dataset[col]]).T
            if xs is not None:
                xs = np.concatenate((xs, feature), axis=1)
            else:
                xs = feature

        trajects = []

        dynamic_feature_len = len(DYNAMIC_FEATURES[self.arg.feature_set])
        for i in range(169-25+1):
            b = [np.array(xs[:,0]).reshape(-1,1)] + [xs[:, j*169+1+i:j*169+1+25+i] for j in range(dynamic_feature_len)]
            c = np.concatenate([b[0]] + [b[i] for i in range(1,dynamic_feature_len+1)], axis=1)
            trajects.append(c)

        xs = np.array(trajects)

        return ids, xs, ys, used_cols

if __name__ == '__main__':
    import argparse
    from visdom import Visdom
    viz = Visdom(port=40077, username='steve', password='visdom0819')
    env = 'data_check'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir'         , type=str)
    parser.add_argument('--feature_set'         , type=str)
    parser.add_argument('--exp_id'              , type=int)


    ARG = parser.parse_args()

    ARG.dataset_dir = '/mnt/aitrics_ext/ext01/steve/working/VitalCare-Model-v2/data/preprocess/SVRC_GW_DTR_P12/preprocess'
    ARG.feature_set = 'LAB'
    ARG.exp_id = 0

    data = Dataset(ARG)
    train_xs = data.train_xs
    train_dynamics = train_xs[:,1:]
    valid_xs = data.valid_xs
    valid_dynamics = valid_xs[:,1:]
    trajec_xs = data.trajectory_xs
    print(trajec_xs.shape)
    # trajec_dynamics = trajec_xs[:,1:]

    # train_dynamics_reshape = train_dynamics.reshape(train_dynamics.shape[0], 17, 25)
    # valid_dynamics_reshape = valid_dynamics.reshape(valid_dynamics.shape[0], 17, 25)
    # trajec_dynamics_reshape = trajec_dynamics.reshape(trajec_dynamics.shape[0], 17, 169)

    # viz.line(train_dynamics_reshape[0,0,:], env=env)
    # viz.line(valid_dynamics_reshape[0,0,:], env=env)
    # viz.line(trajec_dynamics_reshape[0,0,:], env=env)

    for i in np.arange(trajec_xs.shape[0])[::-1]:
        data = trajec_xs[i,0,1:].reshape(-1,25)
        # print(data.shape)
        data = np.transpose(data, [1,0])

        viz.line(data)

        if i < 100:
            break
