import os
import numpy as np

from control.config import args


class experiment_results_validation:
    def __init__(self, args):
        self.args = args
        
        if args.predict_type == "binary":
            self.auc_final_list = []
            self.apr_final_list = []
            self.f1_final_list = []
            self.tpr_final_list = []
            self.tnr_final_list = []
        
        elif "multi_task" in args.predict_type: # multi_task_within, multi_task_range
            self.auc_final_list = []
            self.apr_final_list = []
            self.f1_final_list = []
            self.tpr_final_list = []
            self.tnr_final_list = []
            
            self.all_auc_final_list = []
            self.all_apr_final_list = []
            self.all_f1_final_list = []
            self.all_tpr_final_list = []
            self.all_tnr_final_list = []

        elif "seq_pretrain" == args.predict_type:
            self.train_loss = []
            self.NCE = []

        else:
            exit(1)


    def results_all_seeds(self, list_of_test_results_per_seed):
        if args.predict_type == "binary":
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
        
        elif "multi_task" in args.predict_type:
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
            
            self.all_auc_final_list.append(result[3])
            self.all_apr_final_list.append(result[4])
            self.all_f1_final_list.append(result[5])
            self.all_tpr_final_list.append(result[6])
            self.all_tnr_final_list.append(result[7])
        
        else:
            exit(1)


    def results_per_cross_fold(self):
        print("##########################################################################################")
        if args.predict_type == "binary":
            os.system("echo  \'{} fold cross validation results each fold with {} seeds\'".format(str(self.args.seed_list), str(len(self.args.seed_list))))
            os.system("echo  \'Total mean average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.mean(self.auc_final_list)), 
                str(np.mean(self.apr_final_list)), 
                str(np.mean(self.f1_final_list)), 
                str(np.mean(self.tpr_final_list)), 
                str(np.mean(self.tnr_final_list))))

            os.system("echo  \'Total mean std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.std(self.auc_final_list)), 
                str(np.std(self.apr_final_list)), 
                str(np.std(self.f1_final_list)), 
                str(np.std(self.tpr_final_list)), 
                str(np.std(self.tnr_final_list))))
            
        elif "multi_task" in args.predict_type:
            os.system("echo  \'{} fold cross validation results each fold with {} seeds\'".format(str(self.args.seed_list), str(len(self.args.seed_list))))
            os.system("echo  \'Total mean val average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.mean(self.auc_final_list)), 
                str(np.mean(self.apr_final_list)), 
                str(np.mean(self.f1_final_list)), 
                str(np.mean(self.tpr_final_list)), 
                str(np.mean(self.tnr_final_list))))

            os.system("echo  \'Total mean std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.std(self.auc_final_list)), 
                str(np.std(self.apr_final_list)), 
                str(np.std(self.f1_final_list)), 
                str(np.std(self.tpr_final_list)), 
                str(np.std(self.tnr_final_list))))
            
            all_auc_final_list_np = np.array(self.all_auc_final_list)
            all_apr_final_list_np = np.array(self.all_apr_final_list)
            all_f1_final_list_np = np.array(self.all_f1_final_list)
            all_tpr_final_list_np = np.array(self.all_tpr_final_list)
            all_tnr_final_list_np = np.array(self.all_tnr_final_list)
            
            for intv in range(12):
                if "multi_task_range" in self.args.trainer:
                    msg = f"echo  \'Pred_time: {str(intv)}~{str(intv+1)} "
                else:
                    msg = f"echo  \'Pred_time: 0~{str(intv+1)} "
                
                msg_mean = msg + f"means || auc: {str(np.round(np.mean(all_auc_final_list_np[:,intv]), 4))}, " +\
                                 f"apr: {str(np.round(np.mean(all_apr_final_list_np[:,intv]), 4))}, " +\
                                 f"f1_score: {str(np.round(np.mean(all_f1_final_list_np[:,intv]), 4))}, " +\
                                 f"tpr: {str(np.round(np.mean(all_tpr_final_list_np[:,intv]), 4))}, " +\
                                 f"tnr: {str(np.round(np.mean(all_tnr_final_list_np[:,intv]), 4))}\'"
                
                msg_std = msg + f"stds || auc: {str(np.round(np.std(all_auc_final_list_np[:,intv]), 4))}, " +\
                                 f"apr: {str(np.round(np.std(all_apr_final_list_np[:,intv]), 4))}, " +\
                                 f"f1_score: {str(np.round(np.std(all_f1_final_list_np[:,intv]), 4))}, " +\
                                 f"tpr: {str(np.round(np.std(all_tpr_final_list_np[:,intv]), 4))}, " +\
                                 f"tnr: {str(np.round(np.std(all_tnr_final_list_np[:,intv]), 4))}\'"

                os.system(msg_mean)
                os.system(msg_std)
        
        else:
            exit(1)

class experiment_results:
    def __init__(self, args):
        self.args = args
        
        if args.predict_type == "binary":
            self.auc_final_list = []
            self.apr_final_list = []
            self.f1_final_list = []
            self.tpr_final_list = []
            self.tnr_final_list = []
        
        elif "multi_task" in args.predict_type: # multi_task_within, multi_task_range
            self.auc_final_list = []
            self.apr_final_list = []
            self.f1_final_list = []
            self.tpr_final_list = []
            self.tnr_final_list = []
            
            self.all_auc_final_list = []
            self.all_apr_final_list = []
            self.all_f1_final_list = []
            self.all_tpr_final_list = []
            self.all_tnr_final_list = []
        
        elif "seq_pretrain" == args.predict_type:
            self.train_loss = []
            self.NCE = []
            
        else:
            exit(1)


    def results_all_seeds(self, list_of_test_results_per_seed):
        if args.predict_type == "binary":
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
        
        elif "multi_task" in args.predict_type:
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
            
            self.all_auc_final_list.append(result[3])
            self.all_apr_final_list.append(result[4])
            self.all_f1_final_list.append(result[5])
            self.all_tpr_final_list.append(result[6])
            self.all_tnr_final_list.append(result[7])
        
        else:
            exit(1)


    def results_per_cross_fold(self):
        if args.predict_type == "binary":
            os.system("echo  \'{} fold cross validation results each fold with {} seeds\'".format(str(self.args.seed_list), str(len(self.args.seed_list))))
            os.system("echo  \'Total mean average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.mean(self.auc_final_list)), 
                str(np.mean(self.apr_final_list)), 
                str(np.mean(self.f1_final_list)), 
                str(np.mean(self.tpr_final_list)), 
                str(np.mean(self.tnr_final_list))))

            os.system("echo  \'Total mean std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.std(self.auc_final_list)), 
                str(np.std(self.apr_final_list)), 
                str(np.std(self.f1_final_list)), 
                str(np.std(self.tpr_final_list)), 
                str(np.std(self.tnr_final_list))))
            
        elif "multi_task" in args.predict_type:
            print("##########################################################################################")
            os.system("echo  \'{} fold cross validation results each fold with {} seeds\'".format(str(self.args.seed_list), str(len(self.args.seed_list))))
            os.system("echo  \'Total mean test average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.mean(self.auc_final_list)), 
                str(np.mean(self.apr_final_list)), 
                str(np.mean(self.f1_final_list)), 
                str(np.mean(self.tpr_final_list)), 
                str(np.mean(self.tnr_final_list))))

            os.system("echo  \'Total mean std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.std(self.auc_final_list)), 
                str(np.std(self.apr_final_list)), 
                str(np.std(self.f1_final_list)), 
                str(np.std(self.tpr_final_list)), 
                str(np.std(self.tnr_final_list))))
            
            all_auc_final_list_np = np.array(self.all_auc_final_list)
            all_apr_final_list_np = np.array(self.all_apr_final_list)
            all_f1_final_list_np = np.array(self.all_f1_final_list)
            all_tpr_final_list_np = np.array(self.all_tpr_final_list)
            all_tnr_final_list_np = np.array(self.all_tnr_final_list)
            
            for intv in range(12):
                if "multi_task_range" in self.args.trainer:
                    msg = f"echo  \'Pred_time: {str(intv)}~{str(intv+1)} "
                else:
                    msg = f"echo  \'Pred_time: 0~{str(intv+1)} "
                
                msg_mean = msg + f"means || auc: {str(np.round(np.mean(all_auc_final_list_np[:,intv]), 4))}, " +\
                                 f"apr: {str(np.round(np.mean(all_apr_final_list_np[:,intv]), 4))}, " +\
                                 f"f1_score: {str(np.round(np.mean(all_f1_final_list_np[:,intv]), 4))}, " +\
                                 f"tpr: {str(np.round(np.mean(all_tpr_final_list_np[:,intv]), 4))}, " +\
                                 f"tnr: {str(np.round(np.mean(all_tnr_final_list_np[:,intv]), 4))}\'"
                
                msg_std = msg + f"stds || auc: {str(np.round(np.std(all_auc_final_list_np[:,intv]), 4))}, " +\
                                 f"apr: {str(np.round(np.std(all_apr_final_list_np[:,intv]), 4))}, " +\
                                 f"f1_score: {str(np.round(np.std(all_f1_final_list_np[:,intv]), 4))}, " +\
                                 f"tpr: {str(np.round(np.std(all_tpr_final_list_np[:,intv]), 4))}, " +\
                                 f"tnr: {str(np.round(np.std(all_tnr_final_list_np[:,intv]), 4))}\'"

                os.system(msg_mean)
                os.system(msg_std)
        
        else:
            exit(1)