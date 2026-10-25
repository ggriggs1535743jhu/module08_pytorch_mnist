from utilities import *
import torch
import torchvision
import torchvision.transforms as transforms

def main():

    args = myargs()

    # MNIST dataset
    # Download & load the training dataset, applying a transformation to convert images to PyTorch tensors
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

    # Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)
    #Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)

    # Create an instance of the model & move it to the configured device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)


    # Loss & optimizer
    # CrossEntropyLoss combines nn.LogSoftmax & nn.NLLLoss in one single class
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training the model
    model, loss_list, acc_train, acc_test = training(model, args.num_epochs,
                                                 train_loader, test_loader, criterion, optimizer)
    # Make some plots!
    plot(loss_list, acc_train, acc_test)



if __name__ == "__main__":
    main()