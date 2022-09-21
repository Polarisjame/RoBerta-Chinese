from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score,f1_score,recall_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,StepLR




def train(bert_model, trainloader, testloader, device, learning_rate=0.01, num_epoch=10):
    optimizer = Adam(bert_model.parameters(), learning_rate)  # 使用Adam优化器
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    check_step = 1
    epoch = 1
    lossF = nn.CrossEntropyLoss()
    bert_model.to(device)
    bert_model.train()
    for epoch in range(1, num_epoch + 1):
        # 记录当前epoch的总loss
        total_loss = 0
        # tqdm用以观察训练进度，在console中会打印出进度条

        for step,batch in enumerate(trainloader):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            # 清除现有的梯度
            bert_output = bert_model(inputs)
            # print(bert_output.shape,targets.shape)
            loss = lossF(bert_output, targets)
            loss.backward()
            optimizer.step()
            # 统计总的损失
            total_loss += loss.item()
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss.item()), end="")
        with torch.no_grad():
            testloss = []
            num_correct = 0
            y_true = []
            y_pred = []
            for step,batch in enumerate(testloader):
                inputs, targets = [x.to(device) for x in batch]
                bert_output = bert_model(inputs)
                loss = lossF(bert_output, targets)
                testloss.append(loss.item())
                output = bert_output
                pred = torch.max(output, 1)[1]
                correct_tensor = pred.eq(targets.long().view_as(pred))
                correct = np.squeeze(correct_tensor.cpu().numpy())
                num_correct += np.sum(correct)
                for i,m in zip(pred,targets):
                    y_pred.append(i.cpu())
                    y_true.append(m.cpu())
                rate = (step + 1) / len(testloader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rtest loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss.item()), end="")
            print("Test loss: {:.3f}".format(np.mean(testloss)))
            # accuracy over all test data
            test_acc = num_correct / (len(testloader) * 16)
            y_true=np.array(y_true)
            y_pred=np.array(y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            print('[epoch %d] test_loss: %.3f  test_accuracy: %.3f  test_precission: %.3f  test_recall: %.3f  test_f1-score: %.3f' %
                (epoch, np.mean(testloss), test_acc, precision, recall, f1))
            if epoch%5 == 0:
                torch.save(bert_model,'./share/bert_shop_model.pth')
        scheduler.step()#更新学习率