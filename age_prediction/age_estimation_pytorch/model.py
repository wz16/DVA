import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class NN_outputs_last_linear(nn.Module):
    def __init__(self, pretrain_model, output_dim):
        super(NN_outputs_last_linear,self).__init__()
        self.pretrain_model = pretrain_model
        
        dim_feats = pretrain_model.last_linear.in_features
        self.dim_feats = dim_feats
        self.pretrain_model.last_linear = nn.Identity()
        self.last_layer_new = nn.Linear(dim_feats, output_dim)
        self.pretrain_model.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        last_layer_output = self.pretrain_model(x)
        output = self.last_layer_new(last_layer_output)
        return output,last_layer_output



def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model

def get_model_regress(model_name="se_resnext50_32x4d", num_classes=1, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model

def get_model_regress_uncertainty(model_name="se_resnext50_32x4d", num_classes=3, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model

def get_model_regress_withlastlayer(model_name="se_resnext50_32x4d", num_classes=1, pretrained="imagenet"):
    pretrain_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    wrapped_model = NN_outputs_last_linear(pretrain_model, num_classes)
    return wrapped_model



def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
