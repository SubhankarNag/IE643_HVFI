import sys  
sys.path.append('models')

# best one is = scratch_rl_c_bn_dp2_33i_9f
# available models - 
# "scratch_9i_7f", "finetuned_9i_7f", "finetuned_42i_7f", "finetuned_rl_9i_9f",  "scratch_c_9i_9f",
# "scratch_rl_c_9i_9f", "scratch_rl_c_33i_9f", "scratch_rl_c_bn_dp2_33i_9f", "finetuned_per_9i_7f",
# "finetuned_edge_9i_7f", finetuned_l1_9i_7f

MODEL_NAME = "scratch_rl_c_bn_dp2_33i_9f"
# MODEL_NAME = "finetuned_42i_7f"
print("Model Type =", MODEL_NAME)

if MODEL_NAME == "scratch_9i_7f":
    sys.path.append('models/ECCV_RIFE')
    from ECCV_RIFE.model.RIFE import Model
    pretrained_model_path = './trained_models/scratch_9i_7f/flownet.pkl' 
    
elif MODEL_NAME == "finetuned_9i_7f":
    sys.path.append('models/ECCV_RIFE')
    from ECCV_RIFE.model.RIFE import Model
    pretrained_model_path = './trained_models/finetuned_9i_7f/flownet.pkl' 
    
elif MODEL_NAME == "finetuned_42i_7f":
    sys.path.append('models/ECCV_RIFE')
    from ECCV_RIFE.model.RIFE import Model
    pretrained_model_path = './trained_models/finetuned_42i_7f/flownet.pkl' 
    
elif MODEL_NAME == "scratch_9i_9f":
    sys.path.append('models/ECCV_RIFE')
    from ECCV_RIFE.model.RIFE import Model
    pretrained_model_path = './trained_models/scratch_9i_9f/flownet.pkl'
    
elif MODEL_NAME == "finetuned_rl_9i_9f":
    sys.path.append('models/ECCV_RIFE_region')
    from ECCV_RIFE_region.model.RIFE import Model
    pretrained_model_path = './trained_models/finetuned_rl_9i_9f/flownet.pkl' 
    
elif MODEL_NAME == "scratch_c_9i_9f":
    sys.path.append('models/ECCV_RIFE_context')
    from ECCV_RIFE_context.model.RIFE import Model
    pretrained_model_path = './trained_models/scratch_c_9i_9f/flownet.pkl'
    
elif MODEL_NAME == "scratch_rl_c_9i_9f":
    sys.path.append('models/ECCV_RIFE_C_W')
    from ECCV_RIFE_C_W.model.RIFE import Model
    pretrained_model_path = './trained_models/scratch_rl_c_9i_9f/flownet.pkl' 
    
elif MODEL_NAME == "scratch_rl_c_33i_9f":
    sys.path.append('models/ECCV_RIFE_C_W')
    from ECCV_RIFE_C_W.model.RIFE import Model
    pretrained_model_path = './trained_models/scratch_rl_c_33i_9f/flownet.pkl' 
    
elif MODEL_NAME == "scratch_rl_c_bn_dp2_33i_9f":
    sys.path.append('models/ECCV_RIFE_C_W_BN_DP2')
    from ECCV_RIFE_C_W_BN_DP2.model.RIFE import Model
    pretrained_model_path = './trained_models/scratch_rl_c_bn_dp2_33i_9f/flownet.pkl' 
    
elif MODEL_NAME == "finetuned_per_9i_7f":
    sys.path.append('models/ECCV_RIFE_per')
    from ECCV_RIFE_per.model.RIFE import Model
    pretrained_model_path = './trained_models/finetuned_per_9i_7f/flownet.pkl' 

elif MODEL_NAME == "finetuned_edge_9i_7f":
    sys.path.append('models/ECCV_RIFE_edgeloss')
    from ECCV_RIFE_edgeloss.model.RIFE import Model
    pretrained_model_path = './trained_models/finetuned_edge_9i_7f/flownet.pkl' 
    
elif MODEL_NAME == "finetuned_l1_9i_7f":
    sys.path.append('models/ECCV_RIFE_L1')
    from ECCV_RIFE_L1.model.RIFE import Model
    pretrained_model_path = './trained_models/finetuned_l1_9i_7f/flownet.pkl' 
    