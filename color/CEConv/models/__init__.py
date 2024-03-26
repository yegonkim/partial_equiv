from .resnet import ResNet18 as CEResNet18
from .resnet_partial import ResNet18 as CEResNet18_partial
from .resnet_variational import ResNet18 as CEResNet18_variational

from .flops_resnet import ResNet18 as flops_CEResNet18
from .flops_resnet_partial import ResNet18 as flops_CEResNet18_partial
from .flops_resnet_variational import ResNet18 as flops_CEResNet18_variational

from .resnet import ResNet44 as CEResNet44
from .resnet_partial import ResNet44 as CEResNet44_partial
from .resnet_variational import ResNet44 as CEResNet44_variational

from .flops_resnet import ResNet44 as flops_CEResNet44
from .flops_resnet_partial import ResNet44 as flops_CEResNet44_partial
from .flops_resnet_variational import ResNet44 as flops_CEResNet44_variational

from .cnn import CNN, CECNN
from .cnn_partial import CECNN as CECNN_partial
from .cnn_variational import CECNN as CECNN_variational