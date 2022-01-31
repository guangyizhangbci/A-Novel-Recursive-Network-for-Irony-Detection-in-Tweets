import math
import os
import numpy
import torch
from torch.utils.data import DataLoader
from torch.nn import ModuleList
from modules.nn.models import ModelWrapper
#from modules.nn.modelsbert import ModelWrapperBert
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, recall_score, accuracy_score, \
    precision_score, jaccard_similarity_score

from config import TRAINED_PATH, BASE_PATH, DEVICE
from model.params import TASK3_A, TASK3_B

from utils.load_embeddings import load_word_vectors
from dataloaders.task3 import parse, parse_test
from utils.nlp import twitter_preprocess
from modules.nn.dataloading import WordDataset, CharDataset
from logger.training import class_weigths, Checkpoint, EarlyStop, Trainer
from sklearn.model_selection import ParameterGrid

TASK = 'a'
if TASK == "a":
    model_config = TASK3_A
else:
    model_config = TASK3_B
    
X_train, y_train = parse(task=TASK, dataset="train")
X_val, y_val     = parse(task=TASK  , dataset="gold")
#X_test, y_test   = parse_test(task=TASK)


X_train = X_train + X_val
y_train = y_train + y_val



config=model_config
config['token_type'] = 'word'
config['batch_train'] = 32
config['batch_eval'] = 32



param_grid = {'noise_drop':[[0.1, 0.2, 0.2], [0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.4, 0.3, 0.4], [0.4, 0.4, 0.4], [0.05, 0.1, 0.1]],
              'encoder_size':[100, 200], 'encoder_layers':[2, 3, 4]}

#param_grid = {'attention_heads':[1, 5, 10]}
grid = ParameterGrid(param_grid)

counter = 0
for params in grid:
    counter += 1
    config['embed_noise']     = params['noise_drop'][0]
    config['embed_dropout']   = params['noise_drop'][1]
    config['encoder_dropout'] = params['noise_drop'][2]
    
    config['encoder_size']    = params['encoder_size']
    config['encoder_layers']  = params['encoder_layers']

    def main():
        task = "clf"
        datasets = {
            "train": (X_train, y_train),
            "gold": (X_val, y_val),
        #    "test": (X_test, y_test)
        }
        
        
        name="_".join([model_config["name"], model_config["token_type"]])
        monitor="gold"
        ordinal=False
        pretrained=None
        finetune=None
        label_transformer=None
        disable_cache=False
        if task == "bclf":
            task = "clf"
        
        pretrained_models = None
        pretrained_config = None
        
        ########################################################################
        # Load embeddings
        ########################################################################
        word2idx = None
        if config["token_type"] == "word":
            word_vectors = os.path.join(BASE_PATH, "embeddings", "{}.txt".format(config["embeddings_file"]))
            word_vectors_size = config["embed_dim"]
        
            # load word embeddings
            print("loading word embeddings...")
            word2idx, idx2word, embeddings = load_word_vectors(word_vectors, word_vectors_size)
        
        
        
        ########################################################################
        # DATASET
        # construct the pytorch Datasets and Dataloaders
        ########################################################################
        train_batch_size=config["batch_train"]
        eval_batch_size=config["batch_eval"]
        token_type=config["token_type"]
        params=None if disable_cache else name
        preprocessor=None
        
        if params is not None:
            name = "_".join(params) if isinstance(params, list) else params
        else:
            name = None
        
        loaders = {}
        if token_type == "word":
            
            if word2idx is None:
                raise ValueError
        
            if preprocessor is None:
                preprocessor = twitter_preprocess()
        
            print("Building word-level datasets...")
            for k, v in datasets.items():
                _name = "{}_{}".format(name, k)
                dataset = WordDataset(v[0], v[1], word2idx, name=_name, preprocess=preprocessor, label_transformer=label_transformer)
                batch_size = train_batch_size if k == "train" else eval_batch_size
                loaders[k] = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
                
        elif token_type == "char":
            
            print("Building char-level datasets...")
            for k, v in datasets.items():
                _name = "{}_{}".format(name, k)
                dataset = CharDataset(v[0], v[1], name=_name, label_transformer=label_transformer)
                batch_size = train_batch_size if k == "train" else eval_batch_size
                loaders[k] = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        else:
            raise ValueError("Invalid token_type.")
        
        
        
        
        ########################################################################
        # MODEL
        # Define the model that will be trained and its parameters
        ########################################################################
        out_size = 1
        if task == "clf":
            classes = len(set(loaders["train"].dataset.labels))
            out_size = 1 if classes == 2 else classes
        elif task == "mclf":
            out_size = len(loaders["train"].dataset.labels[0])
        
        num_embeddings = None
        
        if config["token_type"] == "char":
            num_embeddings = len(loaders["train"].dataset.char2idx) + 1
            embeddings = None
        
        model = ModelWrapper(embeddings=embeddings,
                             out_size=out_size,
                             num_embeddings=num_embeddings,
                             pretrained=pretrained_models,
                             finetune=finetune,
                             **config)
        model.to(DEVICE)
        print(model)
        
        
        if task == "clf":
            weights = class_weigths(loaders["train"].dataset.labels, to_pytorch=True)
        if task == "clf":
            weights = weights.to(DEVICE)
        
        
        
        ########################################################################
        # Loss function and optimizer
        ########################################################################
        if task == "clf":
            if out_size > 2:
                criterion = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
        elif task == "reg":
            criterion = torch.nn.MSELoss()
        elif task == "mclf":
            criterion = torch.nn.MultiLabelSoftMarginLoss()
        else:
            raise ValueError("Invalid task!")
        
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        
        optimizer = torch.optim.Adam(parameters, weight_decay=config["weight_decay"])
        
        
        ########################################################################
        # Trainer
        ########################################################################
        def get_pipeline(task, criterion=None, eval=False):
            """
            Generic classification pipeline
            Args:
                task (): available tasks
                        - "clf": multiclass classification
                        - "bclf": binary classification
                        - "mclf": multilabel classification
                        - "reg": regression
                criterion (): the loss function
                eval (): set to True if the pipeline will be used
                    for evaluation and not for training.
                    Note: this has nothing to do with the mode
                    of the model (eval or train). If the pipeline will be used
                    for making predictions, then set to True.
        
            Returns:
            """
            def pipeline(nn_model, curr_batch):
                # get the inputs (batch)
                inputs, labels, lengths, indices = curr_batch
        
                if task in ["reg", "mclf"]:
                    labels = labels.float()
        
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                lengths = lengths.to(DEVICE)
        
                outputs, attentions = nn_model(inputs, lengths)
        
                if eval:
                    return outputs, labels, attentions, None
        
                if task == "bclf":
                    loss = criterion(outputs.view(-1), labels.float())
                else:
                    loss = criterion(outputs.squeeze(), labels)
        
                return outputs, labels, attentions, loss
        
            return pipeline
        
        def calc_pearson(y, y_hat):
            score = pearsonr(y, y_hat)[0]
            if math.isnan(score):
                return 0
            else:
                return score
            
        def get_metrics(task, ordinal):
            _metrics = {
                "reg": {
                    "pearson": calc_pearson,
                },
                "bclf": {
                    "acc": lambda y, y_hat: accuracy_score(y, y_hat),
                    "precision": lambda y, y_hat: precision_score(y, y_hat, average='macro'),
                    "recall": lambda y, y_hat: recall_score(y, y_hat, average='macro'),
                    "f1": lambda y, y_hat: f1_score(y, y_hat, average='macro'),
                },
                "clf": {
                    "acc": lambda y, y_hat: accuracy_score(y, y_hat),
                    "precision": lambda y, y_hat: precision_score(y, y_hat, average='macro'),
                    "recall": lambda y, y_hat: recall_score(y, y_hat, average='macro'),
                    "f1": lambda y, y_hat: f1_score(y, y_hat, average='macro'),
                },
                "mclf": {
                    "jaccard": lambda y, y_hat: jaccard_similarity_score(numpy.array(y), numpy.array(y_hat)),
                    "f1-macro": lambda y, y_hat: f1_score(numpy.array(y), numpy.array(y_hat), average='macro'),
                    "f1-micro": lambda y, y_hat: f1_score(numpy.array(y), numpy.array(y_hat), average='micro'),
                },
            }
            _monitor = {"reg": "pearson", "bclf": "f1", "clf": "f1", "mclf": "jaccard"}
            _mode = {"reg": "max", "bclf": "max", "clf": "max", "mclf": "max"}
        
            if ordinal:
                task = "reg"
                
            metrics = _metrics[task]
            monitor = _monitor[task]
            mode = _mode[task]
            
            return metrics, monitor, mode
        
        
        if task == "clf":
            pipeline = get_pipeline("bclf" if out_size == 1 else "clf", criterion)
        else:
            pipeline = get_pipeline("reg", criterion)
        
        metrics, monitor_metric, mode = get_metrics(task, ordinal)
        
        checkpoint = Checkpoint(name=name, model=model, model_conf=config,
                                monitor=monitor, keep_best=True, scorestamp=True,
                                metric=monitor_metric, mode=mode,
                                base=config["base"])
        early_stopping = EarlyStop(metric=monitor_metric, mode=mode,
                                   monitor=monitor,
                                   patience=config["patience"])
        
        trainer = Trainer(model=model,
                          loaders=loaders,
                          task=task,
                          config=config,
                          optimizer=optimizer,
                          pipeline=pipeline,
                          metrics=metrics,
                          use_exp=True,
                          inspect_weights=False,
                          checkpoint=checkpoint,
                          early_stopping=early_stopping)
        
        
        
        
        ########################################################################
        # model training
        ########################################################################
        epochs = model_config["epochs"]
        checkpoint=False
        unfreeze=0
        
        def unfreeze_module(module, optimizer):
            for param in module.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': list(module.parameters())})
        
        
        print("Training...")
        for epoch in range(epochs):
            trainer.train()
            trainer.eval()
        
            if unfreeze > 0:
                if epoch == unfreeze:
                    print("Unfreeze transfer-learning model...")
                    subnetwork = trainer.model.feature_extractor
                    if isinstance(subnetwork, ModuleList):
                        for fe in subnetwork:
                            unfreeze_module(fe.encoder, trainer.optimizer)
                            unfreeze_module(fe.attention, trainer.optimizer)
                    else:
                        unfreeze_module(subnetwork.encoder, trainer.optimizer)
                        unfreeze_module(subnetwork.attention, trainer.optimizer)
        
            print()
        
            if checkpoint:
                trainer.checkpoint.check()
        
            if trainer.early_stopping.stop():
                print("Early stopping...")
                break
        
    

        ########################################################################
        # model logging
        ########################################################################
        trainer.log_training(name, model_config["token_type"])
        trainer.my_log_training(str(counter)+'-normal', config)

    main()
    torch.cuda.empty_cache()