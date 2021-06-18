# VQMIVC_VoiceConversion
## Environment
*  python 3.6.10
*  torch 1.3.1
*  chainer 6.0.0
*  apex 0.1
*  numpy 1.19.1
*  kaldiio 2.15.1
*  librosa 0.8.0

## Training and inference:
*  Step1. Data preparation & preprocessing
1. put VCTK corpus under directory: 'Dataset/'
2. training/testing speakers split & feature (mel+lf0) extraction:
		python preprocess.py

*  Step2. model training:
1. use mutual information minimization (MIM):
		python train.py use_CSMI=True use_CPMI=True use_PSMI=True
2. no MIM:
		python train.py use_CSMI=False use_CPMI=False use_PSMI=False 

*  Step3. model testing:
1. put PWG vocoder under directory: 'vocoder/'
2. inference with model trained with MIM:
		python convert.py checkpoint=checkpoints/useCSMITrue_useCPMITrue_usePSMITrue_useAmpTrue/model.ckpt-500.pt
3. inference with model trained without MIM:
		python convert.py checkpoint=checkpoints/useCSMIFalse_useCPMIFalse_usePSMIFalse_useAmpTrue/model.ckpt-500.pt
