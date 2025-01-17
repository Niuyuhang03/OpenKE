import config
from  models import *
import json
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'
con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)	
con.set_alpha(0.001)
con.set_bern(0)
con.set_dimension(100)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
con.set_save_steps(10)
con.set_valid_steps(10)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./WN18RR_checkpoint")
con.set_result_dir("./WN18RR_result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(TransE)
con.train()
