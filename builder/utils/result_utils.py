import os
import numpy as np

from control.config import args


class experiment_results_validation:
    def __init__(self, args):
        self.args = args
        
        self.auc_final_list = []
        self.apr_final_list = []
        self.f1_final_list = []
        self.rmse_final_list = []
        self.aucs_final_list = []
        self.aprs_final_list = []
        self.f1s_final_list = []
        
    def results_all_seeds(self, list_of_test_results_per_seed):
        os.system("echo  \'#######################################\'")
        os.system("echo  \'##### Final validation results per seed #####\'")
        os.system("echo  \'#######################################\'")
        seed, result = list_of_test_results_per_seed
        wresult, nresult = result
        aucs, aprs, f1s = nresult
        if "rmse" in self.args.auxiliary_loss_type:
            wauc, wapr, wf1, rmse = wresult
            self.rmse_final_list.append(rmse)
        else:
            wauc, wapr, wf1 = wresult
            
        os.system("echo  \'##### validation results #####\'")
        if "rmse" in self.args.auxiliary_loss_type:
            os.system("echo  \'seed_case:{} - weighted auc: {}, apr: {}, f1_score: {}, rmse: {}\'".format(str(seed), str(wauc), str(wapr), str(wf1), str(rmse)))
        else:
            os.system("echo  \'seed_case:{} - weighted auc: {}, apr: {}, f1: {}\'".format(str(seed), str(wauc), str(wapr), str(wf1)))
        os.system("echo  \'seed_case:Mean auc: {}, apr: {}, f1: {}\'".format(str(np.mean(aucs)), str(np.mean(aprs)), str(np.mean(f1s))))
        os.system("echo  \'seed_case:Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(str(aucs), str(aprs), str(f1s)))

        self.aucs_final_list.append(aucs)
        self.aprs_final_list.append(aprs)
        self.f1s_final_list.append(f1s)
        
        self.auc_final_list.append(wauc)
        self.apr_final_list.append(wapr)
        self.f1_final_list.append(wf1)
        
    def results_per_cross_fold(self):
        print("##########################################################################################")
        os.system("echo  \'{} fold cross validation results each fold with {} seeds\'".format(str(self.args.seed_list), str(len(self.args.seed_list))))
        
        os.system("echo  \'Total validation average -- weighted auc: {}, apr: {}, f1_score: {}\'".format(
            str(np.mean(self.auc_final_list, axis=0)), 
            str(np.mean(self.apr_final_list, axis=0)), 
            str(np.mean(self.f1_final_list, axis=0)))) 

        os.system("echo  \'Total validation std -- weighted auc: {}, apr: {}, f1_score: {}\'".format(
            str(np.std(self.auc_final_list, axis=0)), 
            str(np.std(self.apr_final_list, axis=0)), 
            str(np.std(self.f1_final_list, axis=0))))
        
        if "rmse" in self.args.auxiliary_loss_type:
            os.system("echo  \'Total validation rmse -- avg: {}, std: {}\'".format(
                str(np.mean(self.rmse_final_list, axis=0)),
                str(np.std(self.rmse_final_list, axis=0))))            
            
        os.system("echo  \'Total validation average Mean auc: {}, apr: {}, f1: {}\'".format(
            str(np.mean(self.aucs_final_list)), 
            str(np.mean(self.aprs_final_list)), 
            str(np.mean(self.f1s_final_list))))
        
        os.system("echo  \'Total validation std Mean auc: {}, apr: {}, f1: {}\'".format(
            str(np.std(self.aucs_final_list)), 
            str(np.std(self.aprs_final_list)), 
            str(np.std(self.f1s_final_list))))

        os.system("echo  \'Total validation average Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(
            str(np.mean(self.aucs_final_list, axis=0)), 
            str(np.mean(self.aprs_final_list, axis=0)), 
            str(np.mean(self.f1s_final_list, axis=0))))
        
        os.system("echo  \'Total validation std Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(
            str(np.std(self.aucs_final_list, axis=0)), 
            str(np.std(self.aprs_final_list, axis=0)), 
            str(np.std(self.f1s_final_list, axis=0))))


class experiment_results_test:
    def __init__(self, args):
        self.args = args
        
        self.auc_final_list = []
        self.apr_final_list = []
        self.f1_final_list = []
        self.rmse_final_list = []        
        self.aucs_final_list = []
        self.aprs_final_list = []
        self.f1s_final_list = []
    
    def results_all_seeds(self, list_of_test_results_per_seed):
        os.system("echo  \'#######################################\'")
        os.system("echo  \'##### Final test results per seed #####\'")
        os.system("echo  \'#######################################\'")
        seed, result = list_of_test_results_per_seed
        wresult, nresult = result
        aucs, aprs, f1s = nresult
        if "rmse" in self.args.auxiliary_loss_type:
            wauc, wapr, wf1, rmse = wresult
            self.rmse_final_list.append(rmse)
        else:
            wauc, wapr, wf1 = wresult
            
        os.system("echo  \'##### test results #####\'")
        if "rmse" in self.args.auxiliary_loss_type:
            os.system("echo  \'seed_case:{} - weighted auc: {}, apr: {}, f1_score: {}, rmse: {}\'".format(str(seed), str(wauc), str(wapr), str(wf1), str(rmse)))
        else:
            os.system("echo  \'seed_case:{} - weighted auc: {}, apr: {}, f1: {}\'".format(str(seed), str(wauc), str(wapr), str(wf1)))
        os.system("echo  \'seed_case:Mean auc: {}, apr: {}, f1: {}\'".format(str(np.mean(aucs)), str(np.mean(aprs)), str(np.mean(f1s))))
        os.system("echo  \'seed_case:Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(str(aucs), str(aprs), str(f1s)))

        self.aucs_final_list.append(aucs)
        self.aprs_final_list.append(aprs)
        self.f1s_final_list.append(f1s)
        
        self.auc_final_list.append(wauc)
        self.apr_final_list.append(wapr)
        self.f1_final_list.append(wf1)

    def results_per_cross_fold(self):
        print("##########################################################################################")
        os.system("echo  \'{} fold cross test results each fold with {} seeds\'".format(str(self.args.seed_list), str(len(self.args.seed_list))))
        
        os.system("echo  \'Total test average -- weighted auc: {}, apr: {}, f1_score: {}\'".format(
            str(np.mean(self.auc_final_list, axis=0)), 
            str(np.mean(self.apr_final_list, axis=0)), 
            str(np.mean(self.f1_final_list, axis=0)))) 

        os.system("echo  \'Total test std -- weighted auc: {}, apr: {}, f1_score: {}\'".format(
            str(np.std(self.auc_final_list, axis=0)), 
            str(np.std(self.apr_final_list, axis=0)), 
            str(np.std(self.f1_final_list, axis=0))))
        
        if "rmse" in self.args.auxiliary_loss_type:
            os.system("echo  \'Total test rmse -- avg: {}, std: {}\'".format(
                str(np.mean(self.rmse_final_list, axis=0)),
                str(np.std(self.rmse_final_list, axis=0))))    
        
        os.system("echo  \'Total test average Mean auc: {}, apr: {}, f1: {}\'".format(
            str(np.mean(self.aucs_final_list)), 
            str(np.mean(self.aprs_final_list)), 
            str(np.mean(self.f1s_final_list))))
        
        os.system("echo  \'Total test std Mean auc: {}, apr: {}, f1: {}\'".format(
            str(np.std(self.aucs_final_list)), 
            str(np.std(self.aprs_final_list)), 
            str(np.std(self.f1s_final_list))))
        
        os.system("echo  \'Total test average Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(
            str(np.mean(self.aucs_final_list, axis=0)), 
            str(np.mean(self.aprs_final_list, axis=0)), 
            str(np.mean(self.f1s_final_list, axis=0))))
        
        os.system("echo  \'Total test std Each range auc: {}, \n Each range apr: {}, \n Each range f1: {}\'".format(
            str(np.std(self.aucs_final_list, axis=0)), 
            str(np.std(self.aprs_final_list, axis=0)), 
            str(np.std(self.f1s_final_list, axis=0))))
