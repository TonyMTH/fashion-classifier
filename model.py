from torch import nn

model1 = nn.Sequential(nn.Linear(784, 512),
                      nn.ReLU(),
                      #nn.Dropout(0.1),
                      nn.Linear(512, 256),
                      nn.ReLU(),
                      #nn.Dropout(0.1),
                      nn.Linear(256, 500),
                      nn.ReLU(),
                      #nn.Dropout(0.1),
                      nn.Linear(500, 500),
                      nn.ReLU(),
                      #nn.Dropout(0.1),
                      nn.Linear(500, 10),
                      nn.LogSoftmax(dim = 1)
                     )