import os
import scipy.io as sio
from utils import *
from config import config

# 常量
EEG_NAME_LIST = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                     'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                     'P4', 'P8', 'PO4', 'O2']
REGION_LIST = ['prefrontal lobe', 'left temporosphenoid lobe', 'right temporosphenoid lobe',
                   'occipital lobe', 'parietal lobe', 'central', 'central & parietal lobe',
                   'frontal lobe', 'central & frontal lobe']
PREFRONTAL_LOBE_NAME = ['Fp1', 'Fp2', 'AF3', 'AF4']
LEFT_TEMPOROSPHENOID_NAME = ['FC5', 'CP5', 'P7', 'F7', 'T7']
RIGHT_TEMPOROSPHENOID_NAME = ['FC6', 'CP6', 'P8', 'F8', 'T8']
OCCIPITAL_LOBE_NAME = ['O1', 'O2', 'Oz', 'PO3', 'PO4']
PARIETAL_LOBE_NAME = ['P7', 'P3', 'P4', 'P8', 'Pz']
CENTRAL_NAME = ['C3', 'C4', 'Cz']
CENTRAL_PARIETAL_LOBE_NAME = ['CP1', 'CP2', 'CP5', 'CP6']
FRONTAL_LOBE_NAME = ['F7', 'F3', 'F4', 'F8', 'Fz']
CENTRAL_FRONTAL_LOBE_NAME = ['FC1', 'FC2', 'FC5', 'FC6']
REGION_SUBLISTS = [PREFRONTAL_LOBE_NAME, LEFT_TEMPOROSPHENOID_NAME, RIGHT_TEMPOROSPHENOID_NAME,
                    OCCIPITAL_LOBE_NAME, PARIETAL_LOBE_NAME, CENTRAL_NAME, CENTRAL_PARIETAL_LOBE_NAME,
                    FRONTAL_LOBE_NAME, CENTRAL_FRONTAL_LOBE_NAME]


class DEAPLoader:
    def __init__(self, k_neighbor = 3):
        self.kneigh = k_neighbor
        self.cfg = config()
        self.region_select = list()
        for sublist in REGION_SUBLISTS:
            cur_region = list()
            for item in sublist:
                idx = EEG_NAME_LIST.index(item)
                cur_region.append(idx)
            self.region_select.append(cur_region)

        labels = np.load('data/labels.npz')
        self.vlabel, self.alabel = labels['vlabel'], labels['alabel']
        self.num_of_persons, self.num_of_videos = self.vlabel.shape[0], self.vlabel.shape[1]
        self.num_of_samples = self.num_of_persons * self.num_of_videos
        self.vlabel_oneline, self.alabel_oneline = np.reshape(self.vlabel, (self.num_of_samples,)), np.reshape(self.alabel, (self.num_of_samples,))
        self.label = {'Valence': self.vlabel_oneline, 'Arousal': self.alabel_oneline}

    def tensorprod(self, a, b):
        a_mat = np.repeat(a, b.size).reshape(a.size, b.size)
        b_mat = np.repeat(b, a.size).reshape(b.size, a.size).transpose()
        tprod = np.stack((a_mat,b_mat), axis = 2)
        return tprod
    
    def load_line_kernel(self):
        v_biased_avg_line = self.vlabel.mean(axis = 1) - .5
        v_biased_avg_column = self.vlabel.mean(axis = 0) - .5
        a_biased_avg_line = self.alabel.mean(axis = 1) - .5
        a_biased_avg_column = self.alabel.mean(axis = 0) - .5
        # 利用张量积计算Kernel Feature
        v_ft = self.tensorprod(v_biased_avg_line, v_biased_avg_column)
        a_ft = self.tensorprod(a_biased_avg_line, a_biased_avg_column)
        v_ft, a_ft = np.reshape(v_ft, (self.num_of_samples, 2)), np.reshape(a_ft, (self.num_of_samples, 2))
        ft = {'Valence':v_ft, 'Arousal':a_ft}
        vH, aH = construct_H_with_line_KNN(v_ft, self.kneigh), construct_H_with_line_KNN(a_ft, self.kneigh)
        H = {'Valence':vH, 'Arousal':aH}
        return ft, H

    def load_onehot(self):
        ft = np.zeros((self.num_of_persons, self.num_of_videos, self.num_of_persons + self.num_of_videos))
        for pid in range(self.num_of_persons):
            for vid in range(self.num_of_videos):
                ft[pid][vid][pid] = 1
                ft[pid][vid][self.num_of_persons + vid] = 1
        ft = np.reshape(ft, (ft.shape[0] * ft.shape[1], ft.shape[2]))
        H = construct_H_with_KNN(ft, self.kneigh, False)
        H = {'Valence': H, 'Arousal': H}
        ft = {'Valence': ft, 'Arousal': ft}
        return ft, H

    def load_DEAP_eeg(self, freq=0, channels=list(range(32))):
        eeg_ft_path = os.path.join(self.cfg.feature_folder, 'deap_features_eeg.mat')
        loaded = sio.loadmat(eeg_ft_path)
        eeg_ft = np.array(loaded['deap_feats_eeg'])
        if freq != 0:
            eeg_ft = eeg_ft[:,:,channels,freq-1]
        else:
            eeg_ft = eeg_ft[:,:,channels,:]
        try:
            eeg_ft = np.reshape(eeg_ft, (eeg_ft.shape[0]*eeg_ft.shape[1], eeg_ft.shape[2]*eeg_ft.shape[3]))
        except:
            eeg_ft = np.reshape(eeg_ft, (eeg_ft.shape[0] * eeg_ft.shape[1], eeg_ft.shape[2]))   # for one frequency selection
        eeg_ft_max = eeg_ft.max(axis=0)
        eeg_ft_min = eeg_ft.min(axis=0)
        eeg_ft = (eeg_ft - (eeg_ft_max + eeg_ft_min)/2) / (eeg_ft_max - eeg_ft_min)     # normalization
        eeg_ft *= .0001                                # hyper parameter: 0.0001
        H = construct_H_with_KNN(eeg_ft, self.kneigh)
        H = {'Valence': H, 'Arousal': H}
        eeg_ft = {'Valence': eeg_ft, 'Arousal': eeg_ft}
        return eeg_ft, H

    def load_DEAP_emg(self):
        path_emg = os.path.join(self.cfg.feature_folder, 'deap_features_emg.mat')
        loaded = sio.loadmat(path_emg)
        emg = np.array(loaded['deap_feats_emg'])
        emg = np.reshape(emg, (emg.shape[0]*emg.shape[1], emg.shape[2]*emg.shape[3]))
        emg_max = emg.max(axis=0)
        emg_min = emg.min(axis=0)
        emg = (emg - (emg_max + emg_min)/2) / (emg_max - emg_min)     # normalization
        emg *= .0001
        H = construct_H_with_KNN(emg, self.kneigh)
        emg = {'Valence': emg, 'Arousal': emg}
        H = {'Valence': H, 'Arousal': H}
        return emg, H
    
    def load_DEAP_res(self):
        path_res = os.path.join(self.cfg.feature_folder, 'deap_features_res.mat')
        loaded = sio.loadmat(path_res)
        res = np.array(loaded['deap_feats_res'])
        res = np.reshape(res, (res.shape[0]*res.shape[1], res.shape[2]*res.shape[3]))
        res_max = res.max(axis=0)
        res_min = res.min(axis=0)
        res = (res - (res_max + res_min)/2) / (res_max - res_min)     # normalization
        res *= .0001
        H = construct_H_with_KNN(res, self.kneigh)
        res = {'Valence': res, 'Arousal': res}
        H = {'Valence': H, 'Arousal': H}
        return res, H
    
    def load_DEAP_non_eeg(self):
        path_emg = os.path.join(self.cfg.feature_folder, 'deap_features_emg.mat')
        loaded = sio.loadmat(path_emg)
        emg = np.array(loaded['deap_feats_emg'])
        path_res = os.path.join(self.cfg.feature_folder, 'deap_features_res.mat')
        loaded = sio.loadmat(path_res)
        res = np.array(loaded['deap_feats_res'])
        ft_list = [emg, res]
        for i, ft in enumerate(ft_list):
            ft_list[i] = np.reshape(ft, (ft.shape[0]*ft.shape[1], ft.shape[2]*ft.shape[3]))
            ft_max = ft_list[i].max(axis=0)
            ft_min = ft_list[i].min(axis=0)
            ft_list[i] = (ft_list[i] - (ft_max + ft_min)/2) / (ft_max - ft_min)     # normalization
        non_eeg_ft = np.hstack(ft_list)
        non_eeg_ft *= .0001
        H = construct_H_with_KNN(non_eeg_ft, self.kneigh)
        non_eeg_ft = {'Valence': non_eeg_ft, 'Arousal': non_eeg_ft}
        H = {'Valence': H, 'Arousal': H}
        return non_eeg_ft, H

    def load_DEAP(self, ft_in_use=[0,1,2,3], k_list=[3], freq_in_use=0, region_in_use=0):
        fts, Hs = [], []
        for k in k_list:
            self.kneigh = k
            line_ft, line_H = self.load_line_kernel()
            onehot_ft, onehot_H = self.load_onehot()
            eeg_ft, eeg_H = self.load_DEAP_eeg(freq_in_use,
                                               list(range(32)) if region_in_use==0 else self.region_select[region_in_use-1])
            non_eeg_ft, non_eeg_H = self.load_DEAP_non_eeg()
            fts_k = [line_ft, onehot_ft, eeg_ft, non_eeg_ft]
            Hs_k = [line_H, onehot_H, eeg_H, non_eeg_H]
            fts_k = [fts_k[i] for i in ft_in_use]
            Hs_k = [Hs_k[i] for i in ft_in_use]
            fts = fts + fts_k
            Hs = Hs + Hs_k
        print('dataset successfully loaded')
        return fts, Hs
