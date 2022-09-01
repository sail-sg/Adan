import os
from fairseq.models.roberta import RobertaModel
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef


def get_acc(model_path, data_path, bin_path, task='rte'):
    acc_list = []
    gold, pred = [], []
    roberta = RobertaModel.from_pretrained(
        model_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=bin_path#'RTE-bin'
    )

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    if 'mnli' not in task:
        dev_files = ['dev.tsv']
    else: dev_files = ['dev_mismatched.tsv', 'dev_matched.tsv']
    for dev_file in dev_files:
        with open(os.path.join(data_path, dev_file)) as fin:
            fin.readline()
            for index, line in enumerate(fin):
                tokens = line.strip().split('\t')
                if 'rte' in task or 'qnli' in task:
                    sent1, sent2, target = tokens[1], tokens[2], tokens[3]
                    tokens = roberta.encode(sent1, sent2)
                elif 'qqp' in task:
                    sent1, sent2, target = tokens[3], tokens[4], tokens[5]
                    tokens = roberta.encode(sent1, sent2)
                elif 'mnli' in task:
                    sent1, sent2, target = tokens[8], tokens[9], tokens[11]
                    tokens = roberta.encode(sent1, sent2)
                elif 'mrpc' in task:
                    sent1, sent2, target = tokens[3], tokens[4], tokens[0]
                    tokens = roberta.encode(sent1, sent2)
                elif 'sts_b' in task:
                    sent1, sent2, target = tokens[7], tokens[8], float(tokens[9])
                    tokens = roberta.encode(sent1, sent2)
                elif 'sst_2' in task:
                    sent, target = tokens[0], tokens[1]
                    tokens = roberta.encode(sent)
                   
                elif 'cola' in task:
                    sent, target = tokens[3], tokens[1]
                    tokens = roberta.encode(sent)
                if 'sts_b' not in task:
                    prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
                    prediction_label = label_fn(prediction)
                    ncorrect += int(prediction_label == target)
                    
                    nsamples += 1
                    if 'cola' in task: 
                        target = int(target)
                        prediction_label = int(prediction_label)
                        pred.append(prediction_label)
                        gold.append(target)
                    
                else:
                    features = roberta.extract_features(tokens)
                    predictions = 5.0 * roberta.model.classification_heads['sentence_classification_head'](features)
                    gold.append(target)
                    pred.append(predictions.item())
        if 'cola' in task: 
            out = matthews_corrcoef(gold, pred)
        elif 'sts_b' in task:
            out = pearsonr(gold, pred)[0]
        else: out = float(ncorrect)/float(nsamples)
        
        acc_list.append(out)
    return acc_list


parser = argparse.ArgumentParser(description='GLUE test for acc')
parser.add_argument('--avg_num', type=int, default=1,
                    help='number of try')
parser.add_argument('--pre_path', type=str,  default='./baseline/checkpoint_20_1000000.pt',
                    help='path to pre-trained model')
parser.add_argument('--data_path', type=str,  default='./GLUE/glue_data/STS-B',
                    help='path to data')
parser.add_argument('--bin_path', type=str,  default='./GLUE/STS-B-bin',
                    help='path to -bin data')
parser.add_argument('--finetune_path', type=str,  default='./bert-fintune/adam/STS-B/',
                    help='path to finetuned model')
parser.add_argument('--task', type=str,  default='sts_b',
                    help='task of finetune')
parser.add_argument('--inference', action='store_true', default=False,
                    help='inference only')
args = parser.parse_args()


acc_avg = 0.0
acc_avg2 = 0.0
for _ in range(args.avg_num):
    if not args.inference:
        val = os.system(' fairseq-hydra-train --config-dir ./fairseq/examples/roberta/config/finetuning \
                    --config-name {} \
                    task.data={} checkpoint.restore_file={} \
                    checkpoint.save_dir={}'.format(args.task, args.bin_path, args.pre_path, args.finetune_path))
    all_acc = get_acc(args.finetune_path, args.data_path, args.bin_path, args.task)
    acc_avg+=all_acc[0]
    if len(all_acc)>1:
        acc_avg2+=all_acc[1]

if acc_avg2>0:
    print('Mismatched Accuracy1:{},   Matched Accuracy1:{}'.format(float(acc_avg)/float(args.avg_num), float(acc_avg2)/float(args.avg_num)))
else:
    print('AVG Accuracy1:{}'.format(float(acc_avg)/float(args.avg_num)))

                 