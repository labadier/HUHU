# HUrtful HUmour (HUHU): Detection of humour spreading prejudice in Twitter
This Repository contain the baselines for evaluation in HUHU shared task at Iberlef 2023

HUHU shared task focus  on examining the use of humor to express prejudice towards minorities, specifically analyzing Spanish tweets that are prejudicial towards:

- Women and feminists
- LGBTIQ community
- Immigrants and racially discriminated people
- Overweight people


Three subtasks are prposed to evaluate this phenomenom

#### Subtask 1: HUrtful HUmour Detection

The first subtask consists in determining whether a prejudicial tweet is intended to cause humour. For this, the systems were evaluated and ranked employing the F1-measure over the positive class.

#### Subtask 2A: Prejudice Target Detection

Taking into account the minority groups analyzed, the aim of this subtask is to identify the targeted groups on each tweet as a multilabel classification task. The metric employed to this was macro-F1.

#### Subtask 2B: Degree of Prejudice Prediction

The third subtask consists of predicting on a continuous scale from 1 to 5 to evaluate how prejudicial the message is on average among minority groups. Systems were evaluated employing the Root Mean Squared Error. 

## Using our baselines

To reproduce the results from the proposed baselines you can rely on the python code provided on this repository. Two main paradigms are explored, i.e., the use of classic machine learning (ML) models and neural models, specifically transformer-based.
You can use the following command line instruction to train the baselines:

```shell
python Baseline.py -mode $PARADIGM -model $MODEL -task $TASK -tf TRAINING_DATA -vf DEV_DATA -phase train 
```
Here PARADIGM variable is used to specify wheter to train a transformer-based system or a classic ML approach, MODEL is used in the case you were using transformers models to specify a pretrained model from hugging face 🤗 library
TASK is used to specify the task you will be training your system.
To obtain more information and the choices for each variable you can do 
```shell
python Baseline.py --help
```

To make predictions you can use

```shell
python Baseline.py -model  $MODEL -mode $PARADIGM -phase predict -task $TASK -vf DEV_DATA
```

And for evaluating your predictions:

```shell
python Baseline.py -model $MODEL -mode evaluate -gf DEV_DATA -output $PREDS
```

here PREDS is the path of the fine-tuned model wigths and the previously predicted data.

#### Pretrained transformer based models employed

- dccuchile/bert-base-spanish-wwm-cased
- bigscience/bloom-1b1