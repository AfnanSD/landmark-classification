import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
#         size of the input image is size=(224, 224)
#         self.model = nn.Sequential( 

#             # First conv + maxpool + relu 
#             nn.Conv2d(3, 16, 3, padding=1), 
#             nn.MaxPool2d(2, 2), 
#             nn.BatchNorm2d(16), # added for 4th experiement
#             nn.ReLU(), 
#             nn.Dropout2d(dropout), 
#             #(224, 224, 3) -> (112, 112, 16)

#             # Second conv + maxpool + relu 
#             nn.Conv2d(16, 32, 3, padding=1), 
#             nn.MaxPool2d(2, 2), 
#             nn.BatchNorm2d(32), # added for 4th experiement
#             nn.ReLU(), 
#             nn.Dropout2d(dropout), 
#             #(112, 112, 16) -> (56, 56, 32)

#             # Third conv + maxpool + relu 
#             nn.Conv2d(32, 64, 3, padding=1), 
#             nn.MaxPool2d(2, 2), 
#             nn.BatchNorm2d(64), # added for 4th experiement
#             nn.ReLU(), 
#             nn.Dropout2d(dropout), 
#             #(56, 56, 32) -> (28, 28, 64)
            
#             # Forth conv + maxpool + relu 
#             nn.Conv2d(64, 128, 3, padding=1), 
#             nn.MaxPool2d(2, 2), 
#             nn.BatchNorm2d(128), # added for 4th experiement
#             nn.ReLU(), 
#             nn.Dropout2d(dropout), 
#             #(28, 28, 64) -> (14, 14, 128)
            

#             # Flatten feature maps 
#             nn.Flatten(), 

#             # Fully connected layers.
# #             nn.Linear(25088, 12000), 
# #             nn.ReLU(), 
# #             nn.Dropout(dropout), 
# #             nn.Linear(12000, num_classes) 
            
#             # replaced for 5th experiement
#             nn.Linear(25088, 4096),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(4096, 2048),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(2048, num_classes)
#         ) 
        
    
######## new model for expereiment 6 ######## 
# reordered layers in CNN blocks to become :
# Conv -> Batch normalization -> Relu -> Max pooling -> Dropout
# addec one more CNN layer
        
        # size of the input image is size=(224, 224)
        self.model = nn.Sequential( 

            # First conv + maxpool + relu 
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.BatchNorm2d(16), # added for 4th experiement
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout), 
            #(224, 224, 3) -> (112, 112, 16)

            # Second conv + maxpool + relu 
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.BatchNorm2d(32), # added for 4th experiement
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout), 
            #(112, 112, 16) -> (56, 56, 32)

            # Third conv + maxpool + relu 
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), # added for 4th experiement
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout), 
            #(56, 56, 32) -> (28, 28, 64)
            
            # Forth conv + maxpool + relu 
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), # added for 4th experiement
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout), 
            #(28, 28, 64) -> (14, 14, 128)
            
            # added in 6th experiement
            # Fifth conv + maxpool + relu 
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256), # added for 4th experiement
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout), 
            #(14, 14, 128) -> (7, 7, 256)
            

            # Flatten feature maps 
            nn.Flatten(), 
            
            # replaced for 5th experiement
#             nn.Linear(25088, 4096),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(4096, 2048),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(2048, num_classes)
            
            # updated for 6th experiement
            nn.Linear(12544, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
            
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(iter(data_loaders["train"]))

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
