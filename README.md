## VQMIVC: Vector Quantization and Mutual Information-Based Unsupervised Speech Representation Disentanglement for One-shot Voice Conversion (Interspeech 2021)

### [Paper](https://arxiv.org/abs/2106.10132) | [Pre-trained models](https://drive.google.com/file/d/1Flw6Z0K2QdRrTn5F-gVt6HdR9TRPiaKy/view?usp=sharing) | [Demo](https://wendison.github.io/VQMIVC-demo/)

This paper proposes a speech representation disentanglement framework for one-shot voice conversion, which performs conversion across arbitrary speakers with only a single target-speaker utterance for reference. We employ vector quantization with contrastive predictive coding (VQCPC) for content encoding and introduce mutual information (MI) as the correlation metric during training, to achieve proper disentanglement of content, speaker and pitch representations, by reducing their inter-dependencies in an *unsupervised* manner. In doing so, the proposed method can learn effective disentangled speech representations for retaining source linguistic content and intonation variations, while capturing target speaker characteristics. 

<p align="center">
	<img src='./diagram/architecture.png' width=500 >
</p>
<p align="center">
Diagram of the VQMIVC system.
</p>


## TODO
- [ ] Quick start with pre-trained models


## Requirements
Python 3.6 is used, other requirements are listed in 'requirements.txt'

	pip install -r requirements.txt
	
## Training and inference:
*  Step1. Data preparation & preprocessing
1. Put VCTK corpus under directory: 'Dataset/'
2. Training/testing speakers split & feature (mel+lf0) extraction:

		python preprocess.py

*  Step2. model training:
1. Training with mutual information minimization (MIM):
	
		python train.py use_CSMI=True use_CPMI=True use_PSMI=True

3. Training without MIM:
		
		python train.py use_CSMI=False use_CPMI=False use_PSMI=False 

*  Step3. model testing:
1. Put PWG vocoder under directory: 'vocoder/'
2. Inference with model trained with MIM:
		
		python convert.py checkpoint=checkpoints/useCSMITrue_useCPMITrue_usePSMITrue_useAmpTrue/model.ckpt-500.pt
	
3. Inference with model trained without MIM:

		python convert.py checkpoint=checkpoints/useCSMIFalse_useCPMIFalse_usePSMIFalse_useAmpTrue/model.ckpt-500.pt
