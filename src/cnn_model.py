import torchvision.models as models
import torch.nn as nn

def build_cnn_model(pretrained=True, fine_tune=True, num_classes=10):
    weights = None
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        weights = 'DEFAULT'
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(weights=weights)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model

def build_cnn_no_classifier(pretrained=True, fine_tune=True):
    weights = None
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        weights = 'DEFAULT'
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(weights=weights)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Remove the final classification head.
    model.classifier = nn.Identity()  # Remove the classifier
    return model

