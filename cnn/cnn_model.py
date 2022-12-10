import torch
import torch.nn as nn
import torchvision.models as models


class baseline(nn.Module):
    def __init__(self,in_dim, num_classes):
        super(baseline, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.fc1 = nn.Linear(256, 256)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptiveavgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        y = self.classifier(x)
        return y

class resnet(nn.Module):
    def __init__(self, num_classes):
        super(resnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        
        # freeze 
        for p in self.model.parameters():
            p.requires_grad = False

        # unfreeze last 4 layers
        child_ctr = 0
        for c in self.model.children(): 
            if child_ctr > 14:
                for p in c.parameters():
                    p.requires_grad = True
            child_ctr += 1
        
        fc = nn.Linear(self.model.fc.in_features, num_classes)
        nn.init.kaiming_uniform_(fc.weight)

        self.model.fc = fc
    
    def forward(self, x):
        return self.model(x)