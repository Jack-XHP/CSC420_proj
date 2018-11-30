from keras import backend as K
K.set_image_data_format('channels_first')
from keras.models import model_from_json
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd import grad
import matplotlib.pyplot as plt


def load_facenet():
    """
    loads a saved pretrained model from a json file
    :return:
    """
    # load json and create model
    json_file = open('FRmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FRmodel = model_from_json(loaded_model_json)

    # load weights into new model
    FRmodel.load_weights("FRmodel.h5")
    print("Loaded model from disk")

    return FRmodel


def img_to_encoding(img1, model):
    """
    returns 128-dimensional face embedding for input image
    :param img1:
    :param model:
    :return:
    """
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def load_dataset():

    if not os.path.exists('saved_faces/'):
        os.makedirs('saved_faces')

    a = np.load('faces.npy')

    for i in range(a.shape[0]):

        img = a[i][..., ::-1]
        img = cv2.resize(img, (96, 96))
        cv2.imwrite("saved_faces/face_image_"+str(i)+".jpg", img)


def save_embedding():
    model = load_facenet()
    dir = "saved_faces/"
    save_dir = 'embedding/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for imagePath in os.listdir(dir):
        image = cv2.imread(dir + imagePath)
        embed = img_to_encoding(image, model)
        np.save(save_dir + imagePath.split('.')[0], embed)


def k_mean_cluster():
    # load embedding for the dataset
    dir = 'embedding/'
    embeds = np.zeros(128)
    for embedPath in os.listdir(dir):
        embed = np.load(dir + embedPath)
        embeds = np.vstack([embeds, embed[0]])
    embeds = embeds[1:]

    # compute kmean of 6 clusters
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(embeds)
    return kmeans


def invert_index(kmeans):
    embed_dir = 'embedding/'
    inv_index = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    for embedPath in os.listdir(embed_dir):
        embed = np.load(embed_dir + embedPath)
        index = kmeans.predict(embed)[0]
        inv_index[index].append(embedPath.split('.')[0])
    np.save('invert_index', inv_index)
    np.save('words', kmeans.cluster_centers_)


def matching_face(imagePath):
    # load saved invert index and words
    inv_index = np.load('invert_index.npy').item()
    words = np.load('words.npy')
    
    # find max distance between two cluster center
    max_distance = 0
    for i in range(words.shape[0]):
        j = i + 1
        while j < words.shape[0]:
            distance = np.linalg.norm(words[i] - words[j])
            if distance > max_distance:
                max_distance = distance
            j += 1

    # load model and image
    model = load_facenet()
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (96, 96))
    # compute distance between image's embedding and every cluster center
    embed = img_to_encoding(image, model)
    distance = np.linalg.norm(words - embed[0], ord=2, axis=1)

    # find min distance
    word = np.argmin(distance)

    # if the min distance is bigger than half of the max cluster distance,
    # image should be considered as a outlier
    if distance[word] < max_distance:
        images = np.array(inv_index[word])
        embed_dir = 'embedding/'
        # compute similarity of embeddings
        target_embed = np.array([np.load(embed_dir+image.split('.')[0]+'.npy')[0] for image in images])
        sim = np.dot(target_embed, embed[0]) / np.linalg.norm(embed) / np.linalg.norm(target_embed, axis=1)
        # make threshold to be 0.93
        mask = sim > 0.93
        matches = images[mask]
        print("{} is closest to these images(word {}):".format(imagePath, word))
        print(matches)
        return np.concatenate(([imagePath.split('/')[-1]], matches))
    else:
        print("{} doesn't have matching images:".format(imagePath))
        return [imagePath.split('/')[-1]]


def count_class(class_count, dataset):
    for i in range(len(dataset)):
        data, target = dataset[i]
        class_count[target.item()] += 1
    return class_count


def load_fashion_mnist(batch_size, cuda):
    # get train and test dataset
    FashionMNIST_train = datasets.FashionMNIST('~/data',train=True,download=True,
                                               transform=transforms.Compose(
                                                                            [transforms.ToTensor(),
                                                                             transforms.Normalize((0.1307,), (0.3081,))]))
    FashionMNIST_valid = datasets.FashionMNIST('~/data',train=False,download=True,
                                               transform=transforms.Compose(
                                                                            [transforms.ToTensor(),
                                                                             transforms.Normalize((0.1307,), (0.3081,))]))
    # plot bar graph for classes in two datasets
    class_count = np.zeros(10)
    count_class(class_count, FashionMNIST_train)
    count_class(class_count, FashionMNIST_valid)
    plt.bar(range(10), class_count, tick_label=range(10))
    plt.show()
    
    # make dataloader for the datasets
    kwargs = {'num_workers': 6, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(FashionMNIST_train,batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(FashionMNIST_valid,batch_size=1000, shuffle=False, **kwargs)
    return train_loader, valid_loader


class Net(nn.Module):
    def __init__(self, num_classes, dropRates):
        '''
            model has such structure
                1. Conv->ReLU->dropout->Max Pool
                2. Conv->ReLU->dropout->Max Pool
                3. Conv->ReLU->dropout->Max Pool
                4. Fully Connect->softmax
        '''
        super(Net, self).__init__()
        self.dropRates = dropRates
        channels = [1, 10, 20, 40]
        # define convolution layers' channel size and kernel size
        self.conv_layers = nn.ModuleList([nn.Conv2d(channels[i], channels[i+1], 3, padding=1)for i in range(3)])
        self.last_dim = 360
        # define fully connect layer, softmax if implied by using cross entropy loss
        self.fc = nn.Linear(self.last_dim, num_classes)
    
    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = F.relu(layer(x))
            x = F.dropout2d(x, p=self.dropRates[i], training=self.training)
            x = F.max_pool2d(x, 2)
        x = x.view(-1, self.last_dim)
        x = self.fc(x)
        return x


def test_model(loader, model, cuda):
    model.eval()
    loss = 0
    correct = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        # Load data.
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    correct_rate = 100. * correct/ len(loader.dataset)
    avg_loss = loss / len(loader.dataset)
    return avg_loss, correct_rate, correct


def train_model(train_loader, model, optimizer, cuda):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Load data.
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # Process data and take a step.
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def run_model(batch_size):
    print("Batch size: {}".format(batch_size))
    cuda = torch.cuda.is_available()
    
    # load dataset
    train_loader, valid_loader = load_fashion_mnist(batch_size, cuda)
    train_l = len(train_loader.dataset)
    valid_l = len(valid_loader.dataset)
    
    # define model
    dropout = [0.2, 0.1, 0.1]
    model = Net(10, dropout)
    if cuda:
        model = model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # get initial loss and accuracy
    train_loss, train_correct, train_c = test_model(train_loader, model, cuda)
    valid_loss, valid_correct, valid_c = test_model(valid_loader, model, cuda)
    print("epoch 0: train loss {}, accuracy {} ({}/{})".format(train_loss, train_correct, train_c, train_l))
    print("epoch 0: valid loss {}, accuracy {} ({}/{})".format(valid_loss, valid_correct, valid_c, valid_l))

    # track loss and accuracy along training
    train_losses = [train_loss]
    valid_losses = [valid_loss]
    train_correctes = [train_c]
    valid_correctes = [valid_c]
    epoch = 0

    # stop training when validation loss is not decreasing
    # as it is a signal of overfitting the training set
    while valid_losses[-1] <= valid_losses[epoch-1]:
        epoch += 1
        
        # train model for one epoch
        train_model(train_loader, model, optimizer,cuda)
        
        # get train and validation losses and accuracies
        train_loss, train_correct, train_c = test_model(train_loader, model, cuda)
        valid_loss, valid_correct, valid_c = test_model(valid_loader, model, cuda)
        print("epoch {}: train loss {}, accuracy {} ({}/{})".format(epoch,train_loss, train_correct, train_c, train_l))
        print("epoch {}: valid loss {}, accuracy {} ({}/{})".format(epoch, valid_loss, valid_correct, valid_c, valid_l))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_correctes.append(train_c)
        valid_correctes.append(valid_c)

    # plot training curve for loss and accuarcy
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='Train avg loss')
    plt.plot(epochs, valid_losses, label='Valid avg loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.plot(epochs, np.array(train_correctes) * 100. / train_l, label='Train accuracy')
    plt.plot(epochs, np.array(valid_correctes) * 100. / valid_l, label='Valid accuracy')
    plt.xlabel('epoch')
    plt.ylabel('% accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    load_dataset()
    save_embedding()
    kmeans = k_mean_cluster()
    invert_index(kmeans)
    dir = 'input_faces/'
    f = open('image_matches', 'w')
    for imagePath in os.listdir(dir):
        match = matching_face(dir+imagePath)
        f.write(' '.join(match) + '\n')
    f.close()
    run_model(100)
    x = 8
    for i in range(4):
        run_model(x * 2 ** i)
    
