# QuBayes

This is the official implementation of the paper **Modeling Musical Knowledge with Quantum Bayesian Networks**, presented at the *International Conference on Content-based Multimedia Indexing* in Reykjavik in September 2024 [Krebs2024].

In the code, a discrete Bayesian network is first translated into a Bayesian network with only binary nodes and then translated to a Quantum circuit. Then, queries of the form $P(X | Y)$ can be computed using Quantum Rejection Sampling and the amplitude amplification algorithm as described by [Low2014].


## Setup

```
# Create conda environment
conda create -n qubayes python=3.12
conda activate qubayes
git clone ...
pip install -r requirements.txt
pip install -e .
```

This creates a project with the following structure:

* qubayes/
  * data/
  * qubayes/
    * config.py
    * dataset_stats.py
    * perform_experiment_1.py
    * perform_experiment_2.py
    * qubayes_tools.py
  * tests/
  * README.md
  * requirements.txt
  * setup.py

------------

## Preparation

* Download the [spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) and save the `dataset.csv` file to the `qubayes/data` folder.
* Adjust the paths in the file to match your local environment `qubayes/qubayes/config.py`

## Experiments

To reproduce the results of the paper, execute the following lines of code:

### Experiment 1: Comparison of sampling methods

Execute

```
python perform_experiment1.py -i 5
```

to compute the posterior probabilities mentioned in the paper using classical rejection sampling, quantum rejection sampling, and exact inference.

### Experiment 2: Quantum circuit depth

Execute

```
python perform_experiment2.py
```

to create Figure 4 in the paper, comparing circuit depth and acceptance ratio of a circuit with and without amplitude amplification.

## Errata

Unfortunately, in the paper, Query3 in Table 3 is wrong. The correct query should be:

$P(Artist=Ella Fitzgerald | Genre=Jazz, Mode=major)$, and its exact probability is 0.49.

## References
[Krebs2024]  Krebs, Florian, Hermann Fuerntratt, Roland Unterberger, and Franz Graf. "Modeling Musical Knowledge with Quantum Bayesian Networks." Proceedings of the International Conference on Content-based Multimedia Indexing (2024) 
[Low2014]	 Low, Guang Hao, Theodore J. Yoder, and Isaac L. Chuang. "Quantum inference on Bayesian networks." Physical Review A 89.6 (2014): 062315.



## Citation

If you use this code in your scientific work, we are happy if you cite the following work:

```shell
@InProceedings{krebs2024qbn,
      title={Modeling Musical Knowledge with Quantum Bayesian Networks}, 
      author={Florian Krebs and Hermann Fuerntratt and Roland Unterberger and Franz Graf},
      booktitle={Proceedings of the International Conference on Content-based Multimedia Indexing},
      year={2024}
}
```

