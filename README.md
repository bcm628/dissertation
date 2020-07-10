Dissertation Project
====================
MSc Speech and Language Processing, University of Edinburgh
Domain adaptation as feature extraction for mulitmodal emotion recognition


Code
----

Domain adaptation code based on pytorch_DANN by [CuthbertCai](https://github.com/CuthbertCai/pytorch_DANN "pytorch_DANN"). 
Original code based on [Ganin and Lempitsky](https://arxiv.org/abs/1505.07818 "Domain adversarial training of neural networks"), 2015

Factorized Multimodal Transformer (Zadeh et al, 2019) code from [A2Zadeh](https://github.com/A2Zadeh/Factorized-Multimodal-Transformer "Factorized-Multimodal-Transformer")

publication [here](https://arxiv.org/abs/1911.09826?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29 "Factorized Multimodal Transformer for Multimodal Sequential Learning")

Data
----
This project uses two datasets for domain adaptation: CMU-MOSEI and IEMOCAP.
CMU-MOSEI data is publicly available and can be found [here](http://immortal.multicomp.cs.cmu.edu/). 

In this project, I used the final aligned data from the ACL20 Challenge.

IEMOCAP data is not publicly available and requires permission to be used. I used pre-processed and aligned data from CMU. More information can be found [here](https://github.com/A2Zadeh/CMU-MultimodalSDK "CMU-MultimocalSDK").
