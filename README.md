# Text-Adversarial-Attack
This repo contains the project for the course "Natural Language Processing" of University of Padova. The project consists in implementing 3 different Black-Box Text Adversarial Attacks algorithms, using the library [TextAttack](https://github.com/QData/TextAttack), and evaluating them under various aspects such as attack effectiveness, adversarial semantic coherence, transferability and adversarial training. The considered Adversarial Algorithms are:
- [BAE: BERT-based Adversarial Examples for Text Classification](https://arxiv.org/pdf/2004.01970)
- [BESA: BERT-based Simulated Annealing for Adversarial Text Attacks](https://opus.lib.uts.edu.au/bitstream/10453/168857/2/Bert_SA_IJCAI2021.pdf)
- [Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers](https://arxiv.org/pdf/1801.04354)
  
We performed the experiments in the text classification context againts BERT-based classifiers with high test accuracy downloaded from the HuggingFace hub. The considered datasets, on which the models were pre-trained, are:
- IMDB
- Yelp polarity
- AG News

## Attack Effectiveness, Semantic Similarity and Transferability
To replicate our results, it is sufficient to execute the notebook named `adversarial_attacks.ipynb` from the first to the last cell.

In the BESA part, it may happen that the computation takes a lot of time. In order to overcome this problem, you can split the computation across different machines. The idea is to run one experiment with the first 32 examples on 1 PC, and the other 32 on another machine. You can do this by modifying the code in the Dataset section as follows:

- On the first machine, copy the following code (adapting it for the involved dataset): 
 ```
 imdb = FixedHuggingFaceDataset("imdb", split="test", subset_size=32, shuffle=True)
 ```

- On the second machine:
```
imdb = FixedHuggingFaceDataset("imdb", split="test", subset_size=32, shuffle=True, offset=32)
 ```

Run the experiments in parallel on the two machines and average the results at the end.

DISCLAIMER: If you split the computation, the plots will take into consideration only half of examples.


## Adversarial Training
To replicate our results, it is sufficient to execute the notebooks `adversarial_training_BAEGarg.ipynb` and `adversarial_training_DeepWordBug.ipynb` from the first to the last cell.
