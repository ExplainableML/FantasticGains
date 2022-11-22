### XENT (supervised)
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16

### BYOL
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16

### Barlow
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16

### SimCLR
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16

### VicReg
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name vicreg.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name vicreg.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16

### SimSiam
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simsiam.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simsiam.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16

### MoCo-V3
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mocov3.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mocov3.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16

### SwAV
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name swav.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name swav.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16

### MAE
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++devices=[0,1,2] ++seed=0 ++data.num_workers=16
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++devices=[0,1,2] ++seed=1 ++data.num_workers=16
