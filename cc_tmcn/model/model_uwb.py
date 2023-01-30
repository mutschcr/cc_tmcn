import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
from cc_tmcn.model.common import Conv2d

from pathlib import Path

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(1, 8, kernel_size=(3), padding=1)
        self.conv2 = Conv2d(8, 8, kernel_size=(5), padding=2)
        self.conv3 = Conv2d(8, 8, kernel_size=(15), padding=7)
        self.conv4 = Conv2d(8, 16, kernel_size=(30), padding=15)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 6, 200))
        self.fc1 = nn.Linear(1200, 300)
        self.fc2 = nn.Linear(300, 2)

    def forward_once(self, x):

        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, x):
        
        x1 = self.forward_once(x[0])
        x2 = self.forward_once(x[1])

        return x1, x2

class Model_UWB:
    """
    
    Simple CNN architecture for channel charting
    
    """
 
    def __init__(self, output_directory, name, epochs=200, batch_size=1000):
        """
        Initilization of the model

        Parameters
        ----------
        output_directory : String
            Path to the data used for training. The model is stored within this folder.
        epochs : int
            Number of epochs for training
        batch_size : int
            Batch size
        name : String
            Name of the model.

        Returns
        -------
        None.

        """
        self.model = None
        
        self.epochs = epochs
        self.batch_size = batch_size

        # Path the models are stored while training
        self.model_directory = Path(output_directory) / Path(name)
        self.output_directory = output_directory
        
        self.name = name
        
    def load(self):
        """
        Load a stored model by the model name.

        Returns
        -------
        bool
            True if loading the model is successfull, otherwise False.

        """
        
        print(self.name)

        best_model_path = self.model_directory / Path(self.name + "_best.pt")

        print(best_model_path)

        if best_model_path.exists():
            self.model = self.build_model()
            self.model.load_state_dict(torch.load(self.model_directory / Path(self.name + "_best.pt")))
            return True

        return False

    def predict(self, x_test, use_gpu=True):
        """
        Do a prediction for the given test data

        Parameters
        ----------
        x_test : numpy array
            Input data

        use_gpu : boolean
            True if GPU should be used, false if not (default: True)

        Returns
        -------
        numpy array
            Prediction of the test data

        """

        # Set model to evaluation mode
        self.model.eval()
        if use_gpu:
            self.model.cuda()

        # Restrict the size of the training data sets to 100 sample batches, to avoid memory problems.
        test_loader = torch.utils.data.DataLoader(x_test, batch_size=100,
                                        shuffle=False, num_workers=1)

        result = []
        for _, data in enumerate(test_loader, 0):
            if use_gpu:
                data = data.cuda()
            
            # Predict test data sub set
            result.append(self.model.forward_once(data).cpu().detach().numpy())

            # Free memory on the graphics card
            del data
            if use_gpu:
                torch.cuda.empty_cache()

        return np.concatenate(result)
    
    def get_optimizer(self):
        """
        Gets the optimizer for training

        Returns
        -------
        Optimizer
            Optimizer from torch.optim
        """
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def get_loss(self):        
        """
        Gets the loss function

        Returns
        -------
        Loss function
            lambda as the loss function
        """
        return lambda left, right, dist: torch.mean(torch.abs(dist - torch.norm(left-right, dim=1)))

    def train(self, training_data, use_gpu=True):
        """
        Training of the model

        Parameters
        ----------
        training_data : List
            Training data as list of distance, and input tensors
        use_gpu : boolean
            True if gpu should be used, instead False

        Returns
        -------
        hist : pandas Dataframe
            History of the training process

        """
        # Create model
        if self.model is None:
            self.model = self.build_model()

        if use_gpu:
            # Use GPU
            self.model.cuda()
            print(self.model)

        # Get model path and create folder if not exist
        epochs_dir = self.model_directory / Path("epochs")
        epochs_dir.mkdir(parents=True, exist_ok=True)

        # Batch size and epochs
        batch_size = self.batch_size

        # Setup optimizer
        optimizer = self.get_optimizer()

        # Define loss function
        loss_function = self.get_loss()

        history = pd.DataFrame(columns=['epoch', 'val_loss', 'train_loss'])

        # Repeat optimization for [self.epochs]
        for epoch in range(self.epochs):  
            
            # Set model in training mode
            self.model.train()

            # Get CSI input data for the neural network
            inputs = training_data[1]
            
            # Get indices random combinations of distances
            combinations = np.array([(int(item / len(inputs)), int(item % len(inputs))) for item in np.random.random_integers(0, len(inputs)**2-1, 10000)])

            # Get distances from combination indices
            distances = np.array([training_data[0][item[0], item[1]] for item in combinations])

            # Filter invalid values
            combinations = combinations[~(np.isinf(distances) | np.isnan(distances))]
            distances = distances[~(np.isinf(distances) | np.isnan(distances))]

            # Create Data for the corresponding distances with random batches
            trainloader = torch.utils.data.DataLoader(list(zip(inputs[combinations[:,0]], inputs[combinations[:,1]], distances)), batch_size=batch_size,
                                                    shuffle=False, num_workers=1)

            # Do backproapgation for every batch
            running_loss = 0
            for batch, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels] 
                left_input, right_input, distances = data

                if use_gpu:
                    # use GPU
                    left_input, right_input, distances = left_input.cuda(), right_input.cuda(), distances.cuda()

                # zero the parameter gradients (The optimizer accumulates the gradients, which is only needed for RNNs)
                optimizer.zero_grad()

                # forward pass 
                left_output, right_output = self.model([left_input, right_input])

                # LOSS 
                loss = loss_function(left_output, right_output, distances)

                # set the gradients 
                loss.backward()

                # update weigts w.r.t. the gradients
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                avg_training_loss = running_loss/(batch+1)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch + 1, avg_training_loss), end='\r')
                
                # Delete intermediate results
                del loss
            
            if use_gpu:
                # Empty cuda cache
                torch.cuda.empty_cache()
            
            torch.save(self.model.state_dict(), self.model_directory / Path(self.name + "_best.pt"))

            pd.to_pickle(history, self.model_directory / Path("history.pkl"))
            
            if use_gpu:
                # Empty cuda cache
                torch.cuda.empty_cache()

            print()

        return history

    def build_model(self):
        """
        Builds the model.

        Returns
        -------
        model : Pytorch object
            Pytorch model

        """

        return Net()