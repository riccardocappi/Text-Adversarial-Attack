# Text-Adversarial-Attack

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
