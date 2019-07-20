import  torch
from    torch import optim,nn
import  visdom
import  torchvision
from    torch.utils.data  import DataLoader

from    pokemon import  Pokemon
from    resnet import  ResNet18

batchsz = 32
lr = 1e-3
eporchs = 3
torch.manual_seed(1234)

train_db = Pokemon('pokeman', 224, mode='train')
val_db = Pokemon('pokeman', 224, mode='val')
test_db = Pokemon('pokeman', 224, mode='test')

train_loader = DataLoader(train_db,batch_size=batchsz,shuffle=True,num_workers=4)
val_loader = DataLoader(val_db,batch_size=batchsz,num_workers=2)
test_loader = DataLoader(test_db,batch_size=batchsz,num_workers=2)

viz = visdom.Visdom()

def evalute(model,loader):
    correct = 0
    total = len(loader.dataset)
    for x,y in loader:
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            #print(pred)
        correct += torch.eq(pred,y).sum().float().item() #item 转换到numpy数据类型上去

    return correct / total

def main():
    model = ResNet18(6)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc,best_eporch = 0, 0
    global_step = 0
    viz.line([0],[-1], win='loss', opts=dict(title='loss'))
    viz.line([0],[-1], win='val_acc', opts=dict(title='val_acc'))
    for eporch in range(eporchs):
        for step, (x,y) in enumerate(train_loader):
            #print('x:', x.shape, 'step: ', step)
            #print('y:',y.shape) [32]
            #([32, 3, 224, 224]) step: 23
            #([16, 3, 224, 224]) step: 24   so  total 23*32+16*1 = 768+16=784
            logits = model(x)
            #print('logits', logits.shape) #[32,6]
            loss = criteon(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
        if eporch % 1 == 0:
            val_acc = evalute(model,val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_eporch = eporch

                torch.save(model.state_dict(), 'best.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best_acc', best_acc, 'best_eporch', best_eporch)

    model.load_state_dict(torch.load('best.mdl'))
    print('load from chkpt!!')

    test_acc = evalute(model, test_loader)
    print('test_Acc', test_acc)


if __name__ == '__main__':
    main()
