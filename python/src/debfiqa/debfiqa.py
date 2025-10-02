import torch
import torch.nn as nn


class GradientReverse(torch.autograd.Function):
    '''
    Gradient reverse layer
    Using the factor to control the weight of the reversed gradient
    '''
    @staticmethod
    def forward(ctx, x, factor = 1.0):
        ctx.factor = factor         
        return x.clone()          

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.factor * grad_output, None  

class BasicClassifier(nn.Module):
    '''
        Basic Classifer class for demographic classifiers
    '''
    def __init__(self, input_size = 512, num_classes = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

class AdversarialClassifiers(nn.Module):
    '''
        2 cases:
        (1) use_race = True (DEBFIQA method): race, age and gender classifiers
        (2) use_race = False (DEBFIQAUB/DEBFIQAUBMX method): age and gender classifiers
    '''
    def __init__(self, input_size, use_race = True):
        super().__init__()
        # age
        if use_race:
            self.age_cls = BasicClassifier(input_size = input_size, num_classes = 6)
        else:
            self.age_cls = BasicClassifier(input_size = input_size, num_classes = 2)
        # gender
        self.gender_cls = BasicClassifier(input_size = input_size, num_classes = 2)
        # race
        if use_race:
            self.race_cls = BasicClassifier(input_size = input_size, num_classes = 5)
        
        self.use_race = use_race

    def forward(self, x, factor=0.5):
        #apply the gradient reversal layers
        x_rev = GradientReverse.apply(x, factor)
        if(self.use_race):
            return {
                'age': self.age_cls(x_rev),
                'gender': self.gender_cls(x_rev),
                'race': self.race_cls(x_rev),
            }
        else:
            return {
                'age': self.age_cls(x_rev),
                'gender': self.gender_cls(x_rev),
            }
        
class DebiasedFIQA(nn.Module):
    def __init__(self, backbone, feature_dims, use_race = True):
        super().__init__()
        self.backbone = backbone
        # output: normalized quality scores between [0, 1]
        # try: layernorm or batchnorm1d
        # result: we propose layernorm for relatively small batchsize
        # nn.BatchNorm1d doesn't work for our batchsize
        self.quality_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.PReLU(num_parameters = 256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # adversarial classifiers (parameter: the input dimensions (feature size))
        self.adv_classifiers = nn.ModuleList([
            AdversarialClassifiers(input_size = dim, use_race = use_race) for dim in feature_dims
        ])
        # weights of loss (trainable)
        if use_race:
            self.loss_weights = nn.Parameter(torch.ones(4))
        else:
            self.loss_weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # get multilayer features from network
        features = self.backbone.get_features(x)
        # quality regression using the last feature
        quality = self.quality_regressor(features[-1])
        return features[-1], quality
