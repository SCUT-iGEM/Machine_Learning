from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch

def findFiles(path): return glob.glob(path)

# print(findFiles('data/names/*.txt'))
print('files',findFiles(r'I:\坠机堡垒装不下\E盘Work\SCUT\BioS\iGEM\!\不过过是重头再来\数据数据数据数据\split_txt_data\*.txt'))

import unicodedata
import string

# all_letters = string.ascii_letters + " .,;'"
all_letters = 'ACGT'
print('all_letters',all_letters)
n_letters = len(all_letters)

# 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# print(unicodeToAscii('Ślusàrski'))
print(unicodeToAscii('TGCATTTTTTTCACATCGAGTGCAACTTCGAGAGGCCGGCAGGGTTGGGACACGGACTGCAGGGTTCTCTTATAGCAAAATGATATCGTTCATTGGGGGTTACGGCTGTT'))

# 构建category_lines字典，每种语言的名字列表
category_lines = {}
category_lines_test = {}
all_categories = []

test_num_split = -1000


# 读取文件并分成几行
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# for filename in findFiles('data/names/*.txt'):
for filename in findFiles(r'I:\坠机堡垒装不下\E盘Work\SCUT\BioS\iGEM\!\不过过是重头再来\数据数据数据数据\split_txt_data\*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)[:test_num_split]
    lines_test = readLines(filename)[test_num_split:]
    category_lines[category] = lines            #train_set
    category_lines_test[category] = lines_test  #test_set

n_categories = len(all_categories)
print('all_categories',all_categories)



import torch

# 从all_letters中查找字母索引，例如 "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# 仅用于演示，将字母转换为<1 x n_letters> 张量
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 将一行转换为<line_length x 1 x n_letters>，
# 或一个0ne-hot字母向量的数组
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


print(letterToTensor('A'))

print(lineToTensor('TGCATTTTTTTCACATCGAGTGCAACTTCGAGAGGCCGGCAGGGTTGGGACACGGACTGCAGGGTTCTCTTATAGCAAAATGATATCGTTCATTGGGGGTTACGGCTGTT').size())

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128      #256效果比128和512的好
# n_hidden = 1024       #效果非常差，过拟合，全mid

rnn = RNN(n_letters, n_hidden, n_categories)

input = lineToTensor('TGCATTTTTTTCACATCGAGTGCAACTTCGAGAGGCCGGCAGGGTTGGGACACGGACTGCAGGGTTCTCTTATAGCAAAATGATATCGTTCATTGGGGGTTACGGCTGTT')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)



## 2.train

## 2.1 training prepare function

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def randomTestingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines_test[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

## 2.2 training
criterion = nn.NLLLoss()

# learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
# learning_rate = 0.0005
learning_rate = 0.01  #5个钩
# learning_rate = 0.05  #6个钩  超大数据集上10个钩
# learning_rate = 0.00005
# learning_rate = 0.1     #效果很不好
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]): 
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 将参数的梯度添加到其值中，乘以学习速率
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

import time
import math

# n_iters = 100000    #原始参数
# n_iters = 8632690
# 根据accuracy 跑一半就够了
n_iters = 4000000

# 测试，只跑一点
# n_iters = 100000
# n_iters = 10000

# print_every = 5000
print_every = 5000
# plot_every = 1000
plot_every = 500

# 跟踪绘图的损失
current_loss = 0
all_losses = []

# testset
test_start = 3900000
test_count_yes , test_count_no = 0,0
test_y_true = {}
test_y_pred = {}
test_accuracy = 0
hamming_loss = 0
for cat in all_categories:
    test_y_true[cat]=0
    test_y_pred[cat]=0

def cat_index(cat_name):
    if cat_name == 'very_high':
        return 7
    elif cat_name == 'high':
        return 6
    elif cat_name == 'rare_high':
        return 5
    elif cat_name == 'mid':
        return 4
    elif cat_name == 'rare_low':
        return 3
    elif cat_name == 'low':
        return 2
    elif cat_name == 'very_low':
        return 1

# for iter in range(1, test_n_iters + 1):
#     category, line_t, category_tensor, line_tensor = randomTestingExample()
#     output_t = predict(line_t,line_t,'test')
#     # guess = test(iter,line_t)
#     guess_t,guess_i = categoryFromOutput(output_t)
#     test_y_pred[guess_t]+=1
#     test_y_true[category]+=1
#     if guess_t == category:
#         test_count_yes+=1
#     else:
#         test_count_no+=1
#     test_accuracy = test_count_yes/(test_count_no+test_count_yes)
#     # hamming
#     hamming_loss += cat_index(category)-cat_index(guess_t)
#     # 打印迭代的编号，损失，名字和猜测
#     loss = 0
#     correct = '✓' if guess_t == category else '✗ (%s)' % category
#     print('%d  %d%% (%s) %.4f  %s / %s  %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line_t, guess_t, correct))
#     print('test_accuracy',test_accuracy)

# 多分类评价指标
# Kappa
def compute_kappa(current_n_iters):
    Pe = 0
    for cat in all_categories:
        Pe += test_y_pred[cat]*test_y_true[cat]
    Pe = Pe/(current_n_iters*current_n_iters)
    P0 = test_accuracy
    Kappa = (P0 - Pe)/(1 - Pe)
    return Kappa
# 海明距离
def compute_hamming(current_n_iters):
    hamming_loss_compute = hamming_loss/(current_n_iters)
    return hamming_loss_compute

def timeSince(since):
    now = time.time()
    s = now - since 
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
count_yes=0
count_no=0
# train
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    guess, guess_i = categoryFromOutput(output)
    if guess == category:
        count_yes+=1
    else:
        count_no+=1



    # 打印迭代的编号，损失，名字和猜测
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d  %d%% (%s) %.4f  %s / %s  %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        print('accuracy',count_yes/(count_no+count_yes))

    #加test
    if iter > test_start:
        test_y_pred[guess]+=1
        test_y_true[category]+=1
        if guess == category:
            test_count_yes+=1
        else:
            test_count_no+=1
        # 算总准度
        test_accuracy = test_count_yes/(test_count_no+test_count_yes)
        # 算kappa
        # 算Hamming
        hamming_loss += cat_index(category) - cat_index(guess)
        current_n_iters = iter-test_start
        ham = compute_hamming(current_n_iters)
        kap = compute_kappa(current_n_iters)
        print('hamming',ham)
        print('kappa',kap)



    # 将当前损失平均值添加到损失列表中
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn,'RNN_7split_model_half_data.pt')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

# 在混淆矩阵中跟踪正确的猜测
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# 只需返回给定一行的输出
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 查看一堆正确猜到的例子和记录
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# 通过将每一行除以其总和来归一化
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 设置绘图
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 设置轴
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# 每个刻度线强制标签
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2

def predict(name,input_line,job='nan',n_predictions = 7):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # get 前N类别
        topv, topi = output.topk(n_predictions,1,True)
        predictions = []
        print(name,'start')
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
    if job == 'test':
        return output
def test(name,input_line,n_predictions = 7):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        guess_t , guess_i_t = categoryFromOutput(output)
        # # get 前N类别
        # topv, topi = output.topk(n_predictions,1,True)
        # predictions = []
        # print(name,'start')
        # most_predict = 'nan'
        # most_value = 0
        # for i in range(n_predictions):
        #     value = topv[0][i].item()
        #
        #     category_index = topi[0][i].item()
        #     print('(%.2f) %s' % (value, all_categories[category_index]))
        #     predictions.append([value, all_categories[category_index]])
        #     if value < most_value:
        #         most_value = value
        #         most_predict = all_categories[category_index]
    return guess_t




predict('M1_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCAACGAGGACAGCTAGGTTTATGCAAAAATGTGCCATCACCATAAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAA')
predict('M2_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCGGCGATTCCATCCGTCCGGATGCAAAAATGTGGTTGGAGTTGGAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAA')
predict('M3_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTAAGAGGAGGCCAGGCCGCCGATGCAAAAATGTGGTTGGAGTTGGAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAA')
predict('M4_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTAGCCGTACCGTCGCCCGGGCATGCAAAAATGTGGTTGGAGTTGGAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAA')
predict('M5_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCTCGGGGAGACGGTGCCGTCATGCAAAAATGTGGTTGGAGTTGGAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAA')
predict('M6_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTGGTAGACCTCGGGGCTAAGCATGCAAAAATGTGGTTGGAGTTGGAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAA')
predict('M7_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCTGGCGCGCTGGGCCCTCTCATGCAAAAATGTGGTTGGAGTTGGAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAA')
predict('M8_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCTCGGGGAGACGGTGCCGTCATGCAAACTCGGGGAGACGGTGCCGTCAATGTGGTTGGAGTTGGAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGAT')
predict('M9_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCTCGGGGAGACGGTGCCGTCATGCAAAAATGTGGTTGGAGTTGGAGCTGATTCACTCGGGGAGACGGTGCCGTCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACA')
predict('M10_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCTCGGGGAGACGGTGCCGTCATGCAAACTCGGGGAGACGGTGCCGTCAATGTGGTTGGAGTTGGAGCTGATTCACTCGGGGAGACGGTGCCGTCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGA')
predict('M11_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCTCGGGGAGACGGTGCCGTCATGCAAACTCGGGGAGACGGTGCCGTCAATGTCTCTGTTGGAGTTGGAGAAACAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGAT')
predict('M12_upstream','TTTGATTGATTTGACTGTGTTATTTTGCGTGAGGTTATGAGTAGAAAATAATAATTGAGAAAGGAATATGACAAGAAATATGAAAATAAAGGGAACAAACCCAAATCTGATTGCAAGGAGAGTGAAAGAGCCTTGTTTATATATTTTTTTTTCCTATGTTCTCGGGGAGACGGTGCCGTCATGCAAACTCGGGGAGACGGTGCCGTCAATGTGGTTGGAGTTGGAGCTGATTCAAATGAGCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGAT')


predict('M1_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M2_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M3_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M4_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M5_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M6_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M7_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M8_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M9_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M10_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M11_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')
predict('M12_downstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGTTTAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')


predict('M1_midstream','AGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATCGGCGATTCCATCCGTCCGGTAACAACACAAAATTGTGTTGGAGTTGGAGAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCAGGCAACAAAAATATGCGT')
predict('M2_midstream','TAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGCATAGGTTATGCATTTTGCAAATGC')
predict('M3_midstream','AGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCA')
predict('M4_midstream','AATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGCATAGGTTATGCATTTTGCAAAT')
predict('M5_midstream','AGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGCATAGGTTATGCATTTTGC')
predict('M6_midstream','AAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGCATAGGTTATGCATTTTGCAAA')
predict('M7_midstream','AATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGCATAGGTTATGCATTTTGCAAAT')
predict('M8_midstream','GCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGC')
predict('M9_midstream','AATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGCATAGGTTATGC')
predict('M10_midstream','GAGACGGTGCCGTCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAA')
predict('M11_midstream','GCTAAAAAAAAAATAGTTAGAAAATAAGGTGGTGTTGAACGATAGCAAGTAGATCAAGACACCGTCTAACAGAAAAAGGGGCAGCGGACAATATTATGCAATTATGAAGAAAAGTACTCAAAGGGTCGGAAAAATATTCAAACGATATTTGCATAAAATCCTCAATTGATTGATTATTCCATAGTAAAATACCGTAACAACACAAAATTGTTCTCAAATTCATAAATTATTCATTTTTTCCACGAGCCTCATCACACGAAAAGTCAGAAGAGCATACATAATCTTTTAAATGC')
predict('M12_midstream','TCTTTTAAATGCATAGGTTATGCATTTTGCAAATGCCACCTCTCTGTTGGAGTTGGAGAAAAGCGGGCGGAATCGGGAAGGAAGCCGGAACCACCAAAAACTGGAAGCTACGTTTTTAAGGAAGGTATGGGTGCAGTGTGCTTATCTCAAGAAATATTAGTTATGATATAAGGTGTTGAAGTTTAGAGATAGGTAAATAAACGCGGGGTGTGTTTATTACATGAAGAAGAAGTTAGTTTCTGCCTTGCTTGTTTATCTTGCACATCACATCAGCGGAACATATGCTCACCCAG')



plt.show()




