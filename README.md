# equivaritizer_benchmarks

## Flowers

`python main.py task=flowers model.rot=1`
`python main.py task=flowers model.rot=1 model.insta=True model.insta_params.num_samples=1`
`python main.py task=flowers model.rot=3`
`python main.py task=flowers model.rot=3 model.partial=True`
`python main.py task=flowers model.rot=1 model.insta=True model.insta_params.num_samples=3`
CUDA_VISIBLE_DEVICES=1 nohup sca python main.py task=flowers model.rot=1 &
## MNIST

`python main.py task=mnist model.rot=1`
`python main.py task=mnist model.rot=1 model.insta=True model.insta_params.num_samples=1`
`python main.py task=mnist model.rot=3`
`python main.py task=mnist model.rot=3 model.partial=True`
`python main.py task=mnist model.rot=1 model.insta=True model.insta_params.num_samples=3`

## CIFAR10 Color

`python main.py task=cifar_color no_workers=3 model.rot=1`
`python main.py task=cifar_color no_workers=3 model.rot=1 model.insta=True model.insta_params.num_samples=1`
`python main.py task=cifar_color no_workers=3 model.rot=3`
`python main.py task=cifar_color no_workers=3 model.rot=3 model.partial=True`
`python main.py task=cifar_color no_workers=3 model.rot=1 model.insta=True model.insta_params.num_samples=3`

## CIFAR10 Rotation

T2
`python main.py task=cifar base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4`

Full
4 rotations
`python main.py task=cifar base_group.name=SE2 base_group.no_samples=4 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,1] net.dropout=0 net.final_spatial_dim=[2,2] net.learnable_final_pooling=True net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001`

Partial
4 rotations
`python main.py task=cifar base_group.name=SE2 base_group.no_samples=4 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4`

InstaAug 1 sample
`python main.py task=cifar base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 1]" net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm "net.pool_blocks=[1, 2]" net.type=InstaCKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lamda=0.0001`

Change `basegroup.no_samples` for number of rotations in Full & Partial / number of samples in InstaAug
