#%%

import argparse, sys, os, numpy as np, torch, random, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, f1_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import params
from models import SeqModel, train_model_dev, predict
from pathlib import Path

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def load_data(file_path):
    if os.path.isfile(file_path):
        train = pd.read_csv(file_path).fillna(0)
    else:
        print(f'File {file_path} not found')
        exit(0)

    data = {i: np.array(train[i].to_list()) for i in train.columns}
    if 'mean_prejudice' in data:
        data['mean_prejudice'] = data['mean_prejudice'].reshape(-1, 1)
    return data

def evaluate(gold_file, predictions_path):

    gold = pd.read_csv(gold_file).fillna(0)
    gv_ml = gold['prejudice_woman,prejudice_lgbtiq,prejudice_inmigrant_race,gordofobia'.split(',')].to_numpy()
    gv_humor = gold['humor'].to_numpy()
    gv_prej = gold['mean_prejudice'].to_numpy()

    p_ml, p_humor, p_prejudice = None, None, None
    if os.path.isfile(predictions_path + '_ml.csv'):
        p_ml = pd.read_csv(predictions_path + '_ml.csv').fillna(0)
    if os.path.isfile(predictions_path + '_humor.csv'):
        p_humor = pd.read_csv(predictions_path + '_humor.csv').fillna(0)
    if os.path.isfile(predictions_path + '_mean_prejudice.csv'):
        p_prejudice = pd.read_csv(predictions_path + '_mean_prejudice.csv').fillna(0)

    pv_ml = []
    pv_humor = []
    pv_prej = []

    for index in gold['index'].to_list():
        if p_ml is not None:
            pv_ml += [p_ml[p_ml['index'] == index]['prejudice_woman,prejudice_lgbtiq,prejudice_inmigrant_race,gordofobia'.split(',')].to_numpy()[0]]
        if p_humor is not None:
            pv_humor += [p_humor[p_humor['index'] == index]['humor'].to_numpy()[0]]
        if p_prejudice is not None:
            pv_prej += [p_prejudice[p_prejudice['index'] == index]['mean_prejudice'].to_numpy()[0]]

    if p_humor is not None:
        print('subtask 1 (Humor)',  f1_score(gv_humor, pv_humor))
    if p_ml is not None:
        print('subtask 2 (Target)',  f1_score(gv_ml, pv_ml, average='macro'))
    if p_prejudice is not None:
        print('subtask 3 (Prejudice degree)',  mean_squared_error(gv_prej, pv_prej, squared=False))

    return


def merge_preds(model, output = 'output'):
    
    predictions = {}
    pred_to_save = {'index':[]}
    for i in 'prejudice_woman,prejudice_lgbtiq,prejudice_inmigrant_race,gordofobia'.split(','):
        pred_to_save[i] = []

        if not os.path.isfile(f'output/{model}_{i}.csv'):
            print(f'Predictions for {i} not found')
            return
        
        df = pd.read_csv(f'output/{model}_{i}.csv')
        for _,row in df.iterrows():
            if row['index'] not in predictions:
                predictions[row['index']] = {'index':row['index']}
            predictions[row['index']][i] = row[i]


    for i in predictions:
        for j in pred_to_save:
            pred_to_save[j] += [predictions[i][j]]

    pred_to_save = pd.DataFrame(pred_to_save)
    pred_to_save.to_csv(f'{output}/{model}_ml.csv', index=False)

def check_params(args=None):

    parser = argparse.ArgumentParser(description='Language Model Encoder')

    parser.add_argument('-model', metavar='model', default = params.MODEL, 
    help='Model to be run')
    parser.add_argument('-task', metavar='task', default = None, 
    help='Model to be run', choices='humor,prejudice_woman,prejudice_lgbtiq,prejudice_inmigrant_race,gordofobia,mean_prejudice'.split(','))
    parser.add_argument('-mode', metavar='mode', default = params.MODE, 
    help='Use diferent paradigm learning or evaluate predictions', choices=['evaluate', 'transformer', 'classic'])

    parser.add_argument('-phase', metavar='phase', default = params.PHASE, 
    help='Train evaluate or encode with model', choices=['train', 'predict'])

    parser.add_argument('-output', metavar='output', default = params.OUTPUT, 
    help='Output path for encodings and predictions')
    parser.add_argument('-lr', metavar='lrate', default = params.LR , type=float, 
    help='Learning rate for neural models optimization')
    parser.add_argument('-decay', metavar='decay', default = params.DECAY, type=float,
    help='learning rate decay for neural models optimization')
    parser.add_argument('-interm_layer', metavar='int_layer', default = params.IL, type=int,
    help='amount of intermediate layer neurons')
    parser.add_argument('-epoches', metavar='epoches', default=params.EPOCHES, type=int,
    help='Trainning epoches')
    parser.add_argument('-bs', metavar='batch_size', default=params.BS, type=int,
    help='Batch Size')
    parser.add_argument('-wp', metavar='weigths_path', default=params.OUTPUT,
    help='Saved weights Path')  
    parser.add_argument('-tf', metavar='trainf', default=None,
    help='training data file')
    parser.add_argument('-vf', metavar='valf', default=None,
    help='vaidation data file')
    parser.add_argument('-gf', metavar='goldenf', default=None,
    help='Labeled file for evaluation')

    return parser.parse_args(args)

if __name__ == '__main__':


    parameters = check_params(sys.argv[1:])

    model = parameters.model
    phase = parameters.phase
    mode = parameters.mode
    output = parameters.output
    task = parameters.task

    learning_rate, decay = parameters.lr,  parameters.decay
    interm_layer_size = parameters.interm_layer
    epoches = parameters.epoches
    batch_size = parameters.bs
    gf = parameters.gf
    train_file = parameters.tf
    dev_file = parameters.vf

    weights_path = parameters.wp

    if mode == 'transformer':

        if task is None:
            print('Please specify a task')
            exit(0)

        if phase == 'train':

            Path(output).mkdir(parents=True, exist_ok=True)

            if os.path.exists(output) == False:
                os.system(f'mkdir {output}')

            train = load_data(train_file)
            dev = load_data(dev_file)

            history = None

            history = train_model_dev(model_name=model, data_train=train, data_dev=dev, epoches=epoches, batch_size=batch_size,
            interm_layer_size = interm_layer_size, lr = learning_rate,  decay=decay, output=output, task=task)

        if phase == 'predict':
            modelB = SeqModel(interm_layer_size, model, task)
            modelB.load(os.path.join(weights_path,  f"{model.split('/')[-1]}_{task}"))
            data_dev = load_data(dev_file)

            predict(modelB, model.split('/')[-1], task, data_dev=data_dev)
    
    if mode == 'classic':

        if task is None:
            print('Please specify a task')
            exit(0)

        Path(output).mkdir(parents=True, exist_ok=True)

        if os.path.exists(output) == False:
            os.system(f'mkdir {output}')

        train = load_data(train_file)
        test = load_data(dev_file) # loaded test directly since no parameters are tuned
        vectorizer = TfidfVectorizer(min_df = 0,
                                    max_df = 0.8,
                                    sublinear_tf = True,
                                    analyzer = 'char',
                                    ngram_range=(3, 3), 
                                    use_idf = True)


        train_vectors = vectorizer.fit_transform(train['tweet'])
        test_vectors = vectorizer.transform(test['tweet'])

        if task != 'mean_prejudice':
            modelB = SVC() if model == 'SVM' else RandomForestClassifier()
        else:
            modelB = SVR()

        modelB.fit(train_vectors, train[task])
        pred = modelB.predict(test_vectors)
        
        if task != 'mean_prejudice':
            metrics = classification_report(test[task], pred, target_names=[f'No {task}', task],  digits=4, zero_division=1)        
            print(metrics)
        else:
            metrics = mean_squared_error(test[task], pred, squared=False)
            print(metrics)

        out = {'index': test['index'], task: pred}
        df = pd.DataFrame(out)
        df.to_csv(os.path.join(output, f"{model}_{task}.csv"), index=False)


    if mode == 'evaluate':
        merge_preds(model.split('/')[-1])
        evaluate(gf, f"{output}/{model.split('/')[-1]}")



# %%
