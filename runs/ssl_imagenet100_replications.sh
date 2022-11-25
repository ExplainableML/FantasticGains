# python splitter.py -j jobs.txt -s 1 -tdp /mnt/qb/work/akata/kroth32/Datasets/imagenet-100/train -vdp /mnt/qb/work/akata/kroth32/Datasets/imagenet-100/val --scratch_data --num_gpus 4 --cpus 36 --mem 192
# python splitter.py -j jobs.txt -s 1 -tdp /mnt/qb/akata/shared/ffcv_imagenet100/train_500_0.50_90.ffcv -vdp /mnt/qb/akata/shared/ffcv_imagenet100/val_500_0.50_90.ffcv --scratch_data --num_gpus 4 --cpus 36 --mem 192

### XENT (supervised)
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath

### BYOL
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath

### Barlow
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath

### SimCLR
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath

### VicReg
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name vicreg.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name vicreg.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath

### SimSiam
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simsiam.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simsiam.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath

### MoCo-V3
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mocov3.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mocov3.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath

### SwAV
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name swav.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name swav.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath

### MAE
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath







############ FFCV Replications
### XENT (supervised) - aug-def: default_very_weak
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks_V2"
### BYOL - aug-def: default_weak
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks_V2"
### Barlow - aug-def: default_weak
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks_V2"
### SimCLR - aug-def: default
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks_V2"
### VicReg - aug-def: vicreg
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name vicreg.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks_V2"
### SimSiam - aug_def: default_weak
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simsiam.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks_V2"
### MoCo-V3 - aug-def: default_weak
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mocov3.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks_V2"
### SwAV - aug-def: default
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name swav.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks_V2"
### MAE - aug-def: minimal
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++ffcv_augmentation="default" ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=160 ++wandb.project="IN100_ffcv_ssl_checks_V2"












train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/imagenet-100/train"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/imagenet-100/val"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++devices=[0] ++seed=0 ++data.num_workers=8 ++data.format="dali" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++wandb.enabled=False

# train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenet100/train_500_0.50_90.ffcv"
# val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenet100/val_500_0.50_90.ffcv"
train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/train_320_0.50_90.ffcv"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/val_320_0.50_90.ffcv"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False
# python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False



train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/train_320_0.50_90.ffcv"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/val_320_0.50_90.ffcv"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False


train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/imagenette2-320/train"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/imagenette2-320/val"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="default" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False

python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False




train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/train_320_0.50_90.ffcv"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/val_320_0.50_90.ffcv"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0,1] ++seed=0 ++strategy="ddp" ++precache=True ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False
# python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False




train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/imagenette2-320/train"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/imagenette2-320/val"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0,1] ++seed=0 ++strategy="ddp" ++data.num_workers=8 ++data.format="default" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++precache=True



python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"




python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False




train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/train_320_0.50_90.ffcv"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/val_320_0.50_90.ffcv"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=32
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=32

python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0] ++deterministic=False
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0] ++deterministic=False

train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/train_320_0.50_90.ffcv"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/val_320_0.50_90.ffcv"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0,1] ++deterministic=False
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0,1] ++deterministic=False
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0,1] ++deterministic=False
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name vicreg.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0,1] ++deterministic=False
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simsiam.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0,1] ++deterministic=False
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name swav.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0,1] ++deterministic=False


python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0] ++deterministic=False

#Momentum-based methods struggle with old GPU!
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mocov3.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0,1] ++deterministic=False
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++devices=[0] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="ffcv" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=48 ++devices=[0] ++deterministic=False


train_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/imagenette2-320/train"
val_datapath="/home/karsten_dl/Dropbox/Projects/Datasets/imagenette2-320/val"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1] ++seed=0 ++strategy="dp" ++data.num_workers=8 ++data.format="default" ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++data.dataset="imagenette320" ++checkpoint.enabled=False ++auto_resume.enabled=False ++wandb.enabled=False ++optimizer.batch_size=32




















### XENT (supervised)
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name xent.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"

### BYOL
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name byol.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"

### Barlow
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name barlow.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"

### SimCLR
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simclr.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"

### VicReg
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name vicreg.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name vicreg.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"

### SimSiam
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simsiam.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name simsiam.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"

### MoCo-V3
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mocov3.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mocov3.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"

### SwAV
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name swav.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name swav.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"

### MAE
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++devices=[0,1,2,3] ++seed=0 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
python main_pretrain.py --config-path scripts/replications/imagenet-100 --config-name mae.yaml ++devices=[0,1,2,3] ++seed=1 ++data.num_workers=36 ++data.format="ffcv" ++strategy="dp" ++precache=True ++data.train_path=$train_datapath ++data.val_path=$val_datapath ++optimizer.batch_size=512 ++wandb.project="IN100_ffcv_ssl_checks"
