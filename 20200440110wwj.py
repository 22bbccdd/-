import os
import json
import random
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import paddle
import paddlenlp
import paddle.nn.functional as F
from functools import partial
from paddlenlp.data import Stack, Dict, Pad
from paddlenlp.datasets import load_dataset
import paddle.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from paddlenlp.transformers.auto.tokenizer import AutoTokenizer

seed = 2022
paddle.seed(seed)
random.seed(seed)
np.random.seed(seed)
# 超参数
MODEL_NAME = 'ernie-3.0-base-zh'
# 设置最大阶段长度 和 batch_size
max_seq_length = 365
train_batch_size = 16
valid_batch_size = 16
test_batch_size = 16
# 训练过程中的最大学习率
learning_rate = 8e-5
# 训练轮次
epochs = 50
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01
max_grad_norm = 1.0
# 提交文件名称
sumbit_name = "work/sumbit.csv"
model_logging_dir = 'work/model_logging.csv'
early_stopping = 10
# 是否使用数据增强
enable_dataaug = False
# 是否开启对抗训练
enable_adversarial = False
# Rdrop Loss的超参数，若该值大于0.则加权使用R-drop loss
rdrop_coef = 0.1
# 训练结束后，存储模型参数
save_dir_curr = "checkpoint/{}-{}".format(MODEL_NAME.replace('/','-'),int(time.time()))
def read_jsonfile(file_name):
    data = []
    with open(file_name,encoding='utf-8') as f:
        for i in f.readlines():
            data.append(json.loads(i))
    return data

train = pd.DataFrame(read_jsonfile("data/train.json"))
test = pd.DataFrame(read_jsonfile("data/testA.json"))

print("train size: {} \ntest size {}".format(len(train),len(test)))
train.head(3)
train['text'] = [row['title'] + '，' + row['assignee'] + '，' + row['abstract'] for idx,row in train.iterrows()]
test['text'] = [row['title'] + '，' + row['assignee'] + '，' + row['abstract'] for idx,row in test.iterrows()]
train['concat_len'] = [len(row) for row in train['text']]
test['concat_len'] = [len(row) for row in test['text']]
# 拼接后的文本长度分析
for rate in [0.5,0.75,0.9,0.95,0.99]:
    print("训练数据中{:.0f}%的文本长度小于等于 {:.2f}".format(rate*100,train['concat_len'].quantile(rate)))
plt.title("text length")
sns.distplot(train['concat_len'],bins=10,color='r')
sns.distplot(test['concat_len'],bins=10,color='g')
plt.show()
train_label = train["label_id"].unique()
# 查看标签label分布
plt.figure(figsize=(16,8))
plt.title("label distribution")
sns.countplot(y='label_id',data=train)
# 划分数据集
# 使用留一法划分数据集，训练集：验证集 = 5：1 ，注意这里random_state选择了5，是为了保证36各标签均会出现在训练集和测试集中，某些seed可能会使得验证集的标签不足36个
train_data,valid_data = train_test_split(train,test_size=0.1667,random_state=5)

print("train size: {} \nvalid size {}".format(len(train_data),len(valid_data)))
print("train label: ",sorted(train_data["label_id"].unique()))
print("train label: ",sorted(valid_data["label_id"].unique()))
from paddlenlp.dataaug import WordSubstitute

if enable_dataaug:
    with open("work/data.txt","w") as f:
        for i in train['text']:
            f.write(i+'\n')
        for i in test['text']:
            f.write(i+'\n')

    random.seed(seed)
    np.random.seed(seed)
    tf_idf_file = "work/data.txt"
    aug = WordSubstitute('synonym',
                        tf_idf=True,    # 使用tf-idf
                        tf_idf_file=tf_idf_file,
                        create_n=30,    # 生成增强数据的个数
                        aug_percent=0.15 # 数据增强句子中被替换词数量占全句词比例
                        )

    # 为指定的label生成增强数据
    def data_aug_sample(label_id,data,aug):
        aug_sample = []
        sample = data[data['label_id']==label_id]
        for pre_aug_sample in aug.augment(sample['text']):
            aug_sample.extend(pre_aug_sample)
        return pd.DataFrame({"text":aug_sample,"label_id":[label_id]*len(aug_sample)})

    # 设置每个标签的数据条数
    upper_limit = 180
    # 根据统计信息生成增强数据和采样增强数据
    label_id_indexs = train["label_id"].value_counts().index
    label_id_nums = train["label_id"].value_counts().values
    for label_id,value in zip(label_id_indexs,label_id_nums):
        if value < upper_limit:
            # 计算采样数量
            sample_nums = upper_limit-value
            # 获得增强数据
            label_aug_data = data_aug_sample(data=train_data,label_id=label_id,aug=aug)
            # 如果增强数据的总条数，小于采样数量，将采样数量变为当前增强数据的总条数
            if len(label_aug_data) < sample_nums:
                sample_nums = len(label_aug_data)
            # 采样增强数据
            label_aug_data = label_aug_data.sample(n=sample_nums,random_state=0)
            # 合并到训练集
            train_data = pd.concat((train_data,label_aug_data),axis=0)
            # 重置index
            train_data = train_data.reset_index(drop=True)
    print("train size: {} \nvalid size {}".format(len(train_data),len(valid_data)))
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 创建数据迭代器iter
def read(df,istrain=True):
    if istrain:
        for idx,data in df.iterrows():
            yield {
                "words":data['text'],
                "labels":data['label_id']
                }
    else:
        for idx,data in df.iterrows():
            yield {
                "words":data['text'],
                }

# 将生成器传入load_dataset
train_ds = load_dataset(read, df=train_data, lazy=False)
valid_ds = load_dataset(read, df=valid_data, lazy=False)

# 查看数据
for idx in range(1,3):
    print(train_ds[idx])
    print("==="*30)


# 编码
def convert_example(example, tokenizer, max_seq_len=512, mode='train'):
    # 调用tokenizer的数据处理方法把文本转为id
    tokenized_input = tokenizer(example['words'], is_split_into_words=True, max_seq_len=max_seq_len)
    if mode == "test":
        return tokenized_input
    # 把意图标签转为数字id
    tokenized_input['labels'] = [example['labels']]
    return tokenized_input  # 字典形式，包含input_ids、token_type_ids、labels


train_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='train',
    max_seq_len=max_seq_length)

valid_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='dev',
    max_seq_len=max_seq_length)

# 映射编码
train_ds.map(train_trans_func, lazy=False)
valid_ds.map(valid_trans_func, lazy=False)

# 初始化BatchSampler
train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=train_batch_size, shuffle=True)
valid_batch_sampler = paddle.io.BatchSampler(valid_ds, batch_size=valid_batch_size, shuffle=False)
# print("校准数据是否被seed固定")
# print([*train_batch_sampler][0])
# print([585, 407, 408, 535, 631, 93, 534, 422, 570, 648, 221, 518, 434, 788, 536, 113])

# 定义batchify_fn
batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "labels": Stack(dtype="int32"),
}): fn(samples)

# 初始化DataLoader
train_data_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_sampler=train_batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)

valid_data_loader = paddle.io.DataLoader(
    dataset=valid_ds,
    batch_sampler=valid_batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)

# 相同方式构造测试集
test_ds = load_dataset(read, df=test, istrain=False, lazy=False)

test_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='test',
    max_seq_len=max_seq_length)

test_ds.map(test_trans_func, lazy=False)

test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=test_batch_size, shuffle=False)

test_batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
}): fn(samples)

test_data_loader = paddle.io.DataLoader(
    dataset=test_ds,
    batch_sampler=test_batch_sampler,
    collate_fn=test_batchify_fn,
    return_list=True)
from paddlenlp.transformers.ernie.modeling import ErniePretrainedModel,ErnieForSequenceClassification

class CCFFSLModel(ErniePretrainedModel):
    def __init__(self, ernie, num_classes=2, dropout=None):
        super(CCFFSLModel,self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],num_classes)
        self.apply(self.init_weights)

    def forward(self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None):

        _, pooled_output = self.ernie(input_ids,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
# 创建model
label_classes = train['label_id'].unique()
model = CCFFSLModel.from_pretrained(MODEL_NAME,num_classes=len(label_classes))
# model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME,num_classes=len(label_classes))
# 训练总步数
num_training_steps = len(train_data_loader) * epochs

# 学习率衰减策略
lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(learning_rate, num_training_steps,warmup_proportion)

decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]


# 定义优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params,
    grad_clip=paddle.nn.ClipGradByGlobalNorm(max_grad_norm))


# 验证部分
@paddle.no_grad()
def evaluation(model, data_loader):
    model.eval()
    real_s = []
    pred_s = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        pred_s.extend(probs.argmax(axis=1).numpy())
        real_s.extend(labels.reshape([-1]).numpy())
    score = f1_score(y_pred=pred_s, y_true=real_s, average="macro")
    return score


# 训练阶段
def do_train(model, data_loader):
    print("train ...")
    total_loss = 0.
    model_total_epochs = 0
    best_score = 0.
    num_early_stopping = 0
    if rdrop_coef > 0:
        rdrop_loss = paddlenlp.losses.RDropLoss()
    # 训练
    train_time = time.time()
    valid_time = time.time()
    model.train()
    for epoch in range(0, epochs):
        preds, reals = [], []
        for step, batch in enumerate(data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            # 使用R-drop
            if rdrop_coef > 0:
                logits_2 = model(input_ids=input_ids, token_type_ids=token_type_ids)
                ce_loss = (F.softmax_with_cross_entropy(logits, labels).mean() + F.softmax_with_cross_entropy(logits,
                                                                                                              labels).mean()) * 0.5
                kl_loss = rdrop_loss(logits, logits_2)
                loss = ce_loss + kl_loss * rdrop_coef
            else:
                loss = F.softmax_with_cross_entropy(logits, labels).mean()

            loss.backward()
            # 对抗训练
            if enable_adversarial:
                adv.attack()  # 在 embedding 上添加对抗扰动
                adv_logits = model(input_ids, token_type_ids)
                adv_loss = F.softmax_with_cross_entropy(adv_logits, labels).mean()
                adv_loss.backward()  # 反向传播，并在正常的 grad 基础上，累加对抗训练的梯度
                adv.restore()  # 恢复 embedding 参数

            total_loss += loss.numpy()

            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            model_total_epochs += 1

        #     probs = F.softmax(logits,axis=1)
        #     preds.extend(probs.argmax(axis=1))
        #     reals.extend(labels.reshape([-1]))

        # train_f1 = f1_score(y_pred=preds,y_true=reals,average="macro")
        # print("train f1: %.5f training loss: %.5f speed %.1f s" % (train_f1, total_loss/model_total_epochs,(time.time() - train_time)))
        # train_time = time.time()

        eval_score = evaluation(model, valid_data_loader)
        print("【%.2f%%】validation speed %.2f s" % (
        model_total_epochs / num_training_steps * 100, time.time() - valid_time))
        valid_time = time.time()
        if best_score < eval_score:
            num_early_stopping = 0
            print("eval f1: %.5f f1 update %.5f ---> %.5f " % (eval_score, best_score, eval_score))
            best_score = eval_score
            # 只在score高于0.6的时候保存模型
            if best_score > 0.45:
                # 保存模型
                os.makedirs(save_dir_curr, exist_ok=True)
                save_param_path = os.path.join(save_dir_curr, 'model_best.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                # 保存tokenizer
                tokenizer.save_pretrained(save_dir_curr)
        else:
            num_early_stopping = num_early_stopping + 1
            print("eval f1: %.5f but best f1 %.5f early_stoping_num %d" % (eval_score, best_score, num_early_stopping))
        model.train()
        if num_early_stopping >= early_stopping:
            break
    return best_score
print("best f1 score: %.5f" % best_score)
# logging part
logging_dir = 'work/sumbit'
os.makedirs(logging_dir,exist_ok=True)
logging_name = os.path.join(logging_dir,'run_logging.csv')
os.makedirs(logging_dir,exist_ok=True)

var = [MODEL_NAME, seed, learning_rate, max_seq_length, enable_dataaug, enable_adversarial, rdrop_coef, best_score, save_dir_curr]
names = ['model', 'seed', 'lr', "max_len" , 'enable_dataaug', 'enable_adversarial', 'rdrop_coef','best_score','save_mode_name']
vars_dict = {k: v for k, v in zip(names, var)}
results = dict(**vars_dict)
keys = list(results.keys())
values = list(results.values())

if not os.path.exists(logging_name):
    ori = []
    ori.append(values)
    logging_df = pd.DataFrame(ori, columns=keys)
    logging_df.to_csv(logging_name, index=False)
else:
    logging_df= pd.read_csv(logging_name)
    new = pd.DataFrame(results, index=[1])
    logging_df = logging_df.append(new, ignore_index=True)
    logging_df.to_csv(logging_name, index=False)
# 预测阶段
def do_sample_predict(model,data_loader,is_prob=False):
    model.eval()
    preds = []
    for batch in data_loader:
        input_ids, token_type_ids= batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits,axis=1)
        preds.extend(probs.argmax(axis=1).numpy())
    if is_prob:
        return probs
    return preds

# 读取最佳模型
state_dict = paddle.load(os.path.join(save_dir_curr,'model_best.pdparams'))
model.load_dict(state_dict)

# 预测
print("predict start ...")
pred_score = do_sample_predict(model,test_data_loader)
print("predict end ...")
# 例如sumbit_emtion1.csv 就代表日志index为1的提交结果文件
sumbit = pd.DataFrame({"id":test["id"]})
sumbit["label"] = pred_score
file_name = "work/sumbit/sumbit_fewshot_{}.csv".format(save_dir_curr.split("/")[1])
sumbit.to_csv(file_name,index=False)
print("生成提交文件{}".format(file_name))