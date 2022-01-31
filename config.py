import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:{}".format(DEVICE))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TRAINED_PATH = os.path.join(BASE_PATH, "trained")

EXPS_PATH = os.path.join(BASE_PATH, "out/experiments")

ATT_PATH = os.path.join(BASE_PATH, "out/attentions")

DATA_DIR = BASE_PATH


class TASK1(object):
    pass
#    E_C = {
#        'train': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-train.txt'),
#        'dev': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-dev.txt'),
#        'gold': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-test-gold.txt')
#    }
#
#    EI_oc = {
#        'anger': {
#            'train': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-anger-train.txt'),
#            'dev': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-anger-dev.txt'),
#            'gold': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-anger-test-gold.txt')
#        },
#        'fear': {
#            'train': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-fear-train.txt'),
#            'dev': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-fear-dev.txt'),
#            'gold': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-fear-test-gold.txt')
#        },
#        'sadness': {
#            'train': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-sadness-train.txt'),
#            'dev': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-sadness-dev.txt'),
#            'gold': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-sadness-test-gold.txt')
#        },
#        'joy': {
#            'train': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-joy-train.txt'),
#            'dev': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-joy-dev.txt'),
#            'gold': os.path.join(
#                DATA_DIR, 'task1/EI-oc/EI-oc-En-joy-test-gold.txt')
#        }
#    }
#
#    EI_reg = {
#        'anger': {
#            'train': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-anger-train.txt'),
#            'dev': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-anger-dev.txt'),
#            'gold': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-anger-test-gold.txt')
#        },
#        'fear': {
#            'train': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-fear-train.txt'),
#            'dev': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-fear-dev.txt'),
#            'gold': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-fear-test-gold.txt')
#        },
#        'sadness': {
#            'train': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-sadness-train.txt'),
#            'dev': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-sadness-dev.txt'),
#            'gold': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-sadness-test-gold.txt')
#        },
#        'joy': {
#            'train': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-joy-train.txt'),
#            'dev': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-joy-dev.txt'),
#            'gold': os.path.join(
#                DATA_DIR, 'task1/EI-reg/EI-reg-En-joy-test-gold.txt')
#        }
#    }
#
#    V_oc = {
#        'train': os.path.join(
#            DATA_DIR, 'task1/V-oc/Valence-oc-En-train.txt'),
#        'dev': os.path.join(
#            DATA_DIR, 'task1/V-oc/Valence-oc-En-dev.txt'),
#        'gold': os.path.join(
#            DATA_DIR, 'task1/V-oc/Valence-oc-En-test-gold.txt'),
#    }
#
#    V_reg = {
#        'train': os.path.join(
#            DATA_DIR, 'task1/V-reg/Valence-reg-En-train.txt'),
#        'dev': os.path.join(
#            DATA_DIR, 'task1/V-reg/Valence-reg-En-dev.txt'),
#        'gold': os.path.join(
#            DATA_DIR, 'task1/V-reg/Valence-reg-En-test-gold.txt'),
#    }


class TASK2(object):
    pass
#    EN = os.path.join(DATA_DIR, 'task2/tweet_by_ID_25_10_2017__10_29_45.txt')


class TASK3(object):
    TASK_A = os.path.join(
        DATA_DIR,
        'dataset/2018/datasets/train/SemEval2018-T3-train-taskA_emoji.txt')
    TASK_A_TEST = os.path.join(
        DATA_DIR,
        'dataset/2018/datasets/test_TaskA/SemEval2018-T3_input_test_taskA_emoji.txt')
    TASK_A_GOLD = os.path.join(
        DATA_DIR,
        'dataset/2018/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt')
    TASK_B = os.path.join(
        DATA_DIR,
        'dataset/2018/datasets/train/SemEval2018-T3-train-taskB_emoji.txt')
    TASK_B_TEST = os.path.join(
        DATA_DIR,
        'dataset/2018/datasets/test_TaskB/SemEval2018-T3_input_test_taskB_emoji.txt')
    TASK_B_GOLD = os.path.join(
        DATA_DIR,
        'dataset/2018/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt')



class TASK2019(object):
    TASK_TRAIN = os.path.join(
        DATA_DIR,
        'dataset/2019/training-v1/offenseval-training-v1.tsv')
    
    TASK_TRIAL = os.path.join(
        DATA_DIR,
        'dataset/2019/trial-data/offenseval-trial.txt')
    
    
    TASK_TEST_A = os.path.join(
        DATA_DIR,
        'dataset/2019/Test A Release/testset-taska.tsv')
    
    
    TASK_TEST_B = os.path.join(
        DATA_DIR,
        'dataset/2019/Test B Release/testset-taskb.tsv')
    
    
    TASK_TEST_C = os.path.join(
        DATA_DIR,
        'dataset/2019/Test C Release/test_set_taskc.tsv')