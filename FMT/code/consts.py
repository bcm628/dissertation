from best_metrics import Best


class global_consts():
    single_gpu = True
    load_model = False
    SDK_PATH = "C:/Users/bcmye/PycharmProjects/CMU-MultimodalSDK"

    save_grad = False

    #dataset = "iemocap"
    dataset = 'mosei_new'
    #data_path = "C:/Users/bcmye/PycharmProjects/dissertation/Data/IEMOCAP_aligned"
    data_path = "C:/Users/bcmye/PycharmProjects/CMU-MultimodalSDK/data/MOSEI_aligned"
    model_path = "../model/"

#TODO: change this to not none
    log_path = 'C:/Users/bcmye/PycharmProjects/dissertation/Factorized-Multimodal-Transformer/logs'
    #log_path = None

    HPID = -1

    batch_size = 20

    padding_len = -1

    lr_decay = False

    # cellDim = 150
    # normDim = 100
    # hiddenDim = 300
    config = {
      "cuda": 1,
      "lr": 0.01,
      "epoch_num": 100,
      "dropout": 0.2,
      "seed": 0,
      "gru_lr": 0.001,
      "gru_dropout": 0.2,
      "max_grad": 0.1,

      "n_head": 2,

      "proj_dim_a": 5,

      "proj_dim_v": 20,

      "n_layers": 2,


      "ff_dim_final": 16,

      "dim_total_proj": 8,

      "conv_dims": [3]
    }


    device = None

    best = Best()

    dim_l = -1
    dim_a = -1
    dim_v = -1

    def logParameters(self):
        print( "Hyperparameters:")
        for name in dir(global_consts):
            if name.find("__") == -1 and name.find("max") == -1 and name.find("min") == -1:
                print( "\t%s: %s" % (name, str(getattr(global_consts, name))))
