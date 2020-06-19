# Class order: Anger Disgust Fear Happy Sad Surprise

# python main.py -bs=512 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/cmu-mosei/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=0 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --ckpt='savings/mosei_emo/models/ddd.pt' --test


#####
# Naive fine-tune
#####
## Anger
# Train on 5 emotions
# python main.py -bs=512 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 0

# few-shot (fine-tune) on the sixth emotion (Anger)
# python main.py -bs=8 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --fsl2 0 --ckpt='savings/mosei_emo/models/eea_wacc_0.6579_f1_0.7088_auc_0.7090_ep25_rand0_[300, 200, 100]_tav_bi_zsl0.pt'

# few-shot (fine-tune w/ continual learning) on the sixth emotion (Anger)
# python ft_gem.py -bs=8 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 0 --fsl2 0 --gem --margin 0.01 --ckpt='savings/mosei_emo/models/eea_wacc_0.6579_f1_0.7088_auc_0.7090_ep25_rand0_[300, 200, 100]_tav_bi_zsl0.pt'

# few-shot (fine-tune w/ joint) on the sixth emotion (Anger)
# python ft_gem.py -bs=8 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 0 --fsl2 0 --joint --ckpt='savings/mosei_emo/models/eea_wacc_0.6579_f1_0.7088_auc_0.7090_ep25_rand0_[300, 200, 100]_tav_bi_zsl0.pt'


## Disgust
# Train on 5 emotions
# python main.py -bs=512 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 1

# few-shot (fine-tune) on the sixth emotion (Disgust)
# python main.py -bs=8 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --fsl2 1 --ckpt='savings/mosei_emo/models/eea_wacc_0.6380_f1_0.6782_auc_0.6872_ep17_rand0_[300, 200, 100]_tav_bi_zsl1.pt'

# few-shot (fine-tune w/ continual learning) on the sixth emotion (Disgust)
# python ft_gem.py -bs=16 -lr=3e-5 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 1 --fsl2 1 --gem --margin 0.01 --ckpt='savings/mosei_emo/models/eea_wacc_0.6380_f1_0.6782_auc_0.6872_ep17_rand0_[300, 200, 100]_tav_bi_zsl1.pt'

# few-shot (fine-tune w/ joint) on the sixth emotion (Disgust)
# python ft_gem.py -bs=16 -lr=3e-5 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 1 --fsl2 1 --joint --ckpt='savings/mosei_emo/models/eea_wacc_0.6380_f1_0.6782_auc_0.6872_ep17_rand0_[300, 200, 100]_tav_bi_zsl1.pt'


## Fear
# Train on 5 emotions
# python main.py -bs=512 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 2

# few-shot (fine-tune) on the sixth emotion (Fear)
# python main.py -bs=4 -lr=1e-5 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --fsl2 2 --ckpt='savings/mosei_emo/models/eea_wacc_0.6641_f1_0.6829_auc_0.7134_ep15_rand0_[300, 200, 100]_tav_bi_zsl2.pt'

# few-shot (fine-tune w/ continual learning) on the sixth emotion (Fear)
# python ft_gem.py -bs=16 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 2 --fsl2 2 --gem --margin 0.01 --ckpt='savings/mosei_emo/models/eea_wacc_0.6641_f1_0.6829_auc_0.7134_ep15_rand0_[300, 200, 100]_tav_bi_zsl2.pt'

# few-shot (fine-tune w/ joint) on the sixth emotion (Fear)
# python ft_gem.py -bs=16 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 2 --fsl2 2 --joint --ckpt='savings/mosei_emo/models/eea_wacc_0.6641_f1_0.6829_auc_0.7134_ep15_rand0_[300, 200, 100]_tav_bi_zsl2.pt'


## Happy
# Train on 5 emotions
# python main.py -bs=512 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 3

# few-shot (fine-tune) on the sixth emotion (Happy)
# python main.py -bs=8 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --fsl2 3 --ckpt='savings/mosei_emo/models/eea_wacc_0.6293_f1_0.6675_auc_0.6767_ep17_rand0_[300, 200, 100]_tav_bi_zsl3.pt'

# few-shot (fine-tune w/ continual learning) on the sixth emotion (Happy)
# python ft_gem.py -bs=16 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 3 --fsl2 3 --gem --margin 0.01 --ckpt='savings/mosei_emo/models/eea_wacc_0.6293_f1_0.6675_auc_0.6767_ep17_rand0_[300, 200, 100]_tav_bi_zsl3.pt'

# few-shot (fine-tune w/ joint) on the sixth emotion (Happy)
# python ft_gem.py -bs=16 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 3 --fsl2 3 --joint --ckpt='savings/mosei_emo/models/eea_wacc_0.6293_f1_0.6675_auc_0.6767_ep17_rand0_[300, 200, 100]_tav_bi_zsl3.pt'


## Sad
# Train on 5 emotions
# python main.py -bs=512 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 4

# few-shot (fine-tune) on the sixth emotion (Sad)
# python main.py -bs=8 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --fsl2 4 --ckpt='savings/mosei_emo/models/eea_wacc_0.6643_f1_0.7229_auc_0.7155_ep15_rand0_[300, 200, 100]_tav_bi_zsl4.pt'

# few-shot (fine-tune w/ continual learning) on the sixth emotion (Sad)
# python ft_gem.py -bs=16 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 4 --fsl2 4 --gem --margin 0.01 --ckpt='savings/mosei_emo/models/eea_wacc_0.6643_f1_0.7229_auc_0.7155_ep15_rand0_[300, 200, 100]_tav_bi_zsl4.pt'

# few-shot (fine-tune w/ joint) on the sixth emotion (Sad)
# python ft_gem.py -bs=16 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 4 --fsl2 4 --joint --ckpt='savings/mosei_emo/models/eea_wacc_0.6643_f1_0.7229_auc_0.7155_ep15_rand0_[300, 200, 100]_tav_bi_zsl4.pt'


## Surprise
# Train on 5 emotions
# python main.py -bs=512 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 5

# few-shot (fine-tune) on the sixth emotion (Surprise)
# python main.py -bs=8 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=2 --threshold=0.5 --clip=10.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --fsl2 5 --ckpt='savings/mosei_emo/models/eea_wacc_0.6683_f1_0.7083_auc_0.7224_ep16_rand0_[300, 200, 100]_tav_bi_zsl5.pt'

# few-shot (fine-tune w/ continual learning) on the sixth emotion (Surprise)
# python ft_gem.py -bs=16 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 5 --fsl2 5 --gem --margin 0.01 --ckpt='savings/mosei_emo/models/eea_wacc_0.6683_f1_0.7083_auc_0.7224_ep16_rand0_[300, 200, 100]_tav_bi_zsl5.pt'

# few-shot (fine-tune w/ joint) on the sixth emotion (Surprise)
# python ft_gem.py -bs=16 -lr=1e-4 -ep=100 --aligned --model=eea --data-folder=./data/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=bce --cuda=3 --threshold=0.5 --clip=1 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav -bi --zsl 5 --fsl2 5 --joint --ckpt='savings/mosei_emo/models/eea_wacc_0.6683_f1_0.7083_auc_0.7224_ep16_rand0_[300, 200, 100]_tav_bi_zsl5.pt'