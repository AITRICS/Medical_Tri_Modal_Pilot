import os
import numpy as np

from control.config import args


class experiment_results_validation:
    def __init__(self, args):
        self.args = args
        
        self.auc_final_list = []
        self.apr_final_list = []
        self.f1_final_list = []
        self.tpr_final_list = []
        self.tnr_final_list = []
        
    def results_all_seeds(self, list_of_test_results_per_seed):
        os.system("echo  \'#######################################\'")
        os.system("echo  \'##### Final validation results per seed #####\'")
        os.system("echo  \'#######################################\'")
        seed, result, tpr, tnr = list_of_test_results_per_seed
        os.system("echo  \'seed_case:{} -- auc: {}, apr: {}, f1_score: {}, tpr: {}, tnr: {}\'".\
            format(seed, str(result[0]), str(result[1]), str(result[2]), str(tpr), str(tnr)))
            
        self.auc_final_list.append(result[0])
        self.apr_final_list.append(result[1])
        self.f1_final_list.append(result[2])
        self.tpr_final_list.append(tpr)
        self.tnr_final_list.append(tnr)
        
    def results_per_cross_fold(self):
        print("##########################################################################################")
        os.system("echo  \'{} fold cross validation results each fold with {} seeds\'".format(str(self.args.seed_list), str(len(self.args.seed_list))))
        os.system("echo  \'Total validation average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
            str(np.mean(self.auc_final_list)), 
            str(np.mean(self.apr_final_list)), 
            str(np.mean(self.f1_final_list)), 
            str(np.mean(self.tpr_final_list)), 
            str(np.mean(self.tnr_final_list))))

        os.system("echo  \'Total validation std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
            str(np.std(self.auc_final_list)), 
            str(np.std(self.apr_final_list)), 
            str(np.std(self.f1_final_list)), 
            str(np.std(self.tpr_final_list)), 
            str(np.std(self.tnr_final_list))))

class experiment_results:
    def __init__(self, args):
        self.args = args
        
        self.auc_final_list = []
        self.apr_final_list = []
        self.f1_final_list = []
        self.tpr_final_list = []
        self.tnr_final_list = []
    
    def results_all_seeds(self, list_of_test_results_per_seed):
        os.system("echo  \'#######################################\'")
        os.system("echo  \'##### Final test results per seed #####\'")
        os.system("echo  \'#######################################\'")
        seed, result, tpr, tnr = list_of_test_results_per_seed
        os.system("echo  \'seed_case:{} -- auc: {}, apr: {}, f1_score: {}, tpr: {}, tnr: {}\'".\
            format(seed, str(result[0]), str(result[1]), str(result[2]), str(tpr), str(tnr)))
            
        self.auc_final_list.append(result[0])
        self.apr_final_list.append(result[1])
        self.f1_final_list.append(result[2])
        self.tpr_final_list.append(tpr)
        self.tnr_final_list.append(tnr)

    def results_per_cross_fold(self):
        os.system("echo  \'{} fold cross validation results each fold with {} seeds\'".format(str(self.args.seed_list), str(len(self.args.seed_list))))
        os.system("echo  \'Total test average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
            str(np.mean(self.auc_final_list)), 
            str(np.mean(self.apr_final_list)), 
            str(np.mean(self.f1_final_list)), 
            str(np.mean(self.tpr_final_list)), 
            str(np.mean(self.tnr_final_list))))

        os.system("echo  \'Total test std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
            str(np.std(self.auc_final_list)), 
            str(np.std(self.apr_final_list)), 
            str(np.std(self.f1_final_list)), 
            str(np.std(self.tpr_final_list)), 
            str(np.std(self.tnr_final_list))))
        
        