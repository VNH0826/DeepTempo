# DeepTempo-Learning-Temporal-Circuit-Behaviors-of-Logic-Gates
This repository is the corresponding model repository for the paper " DeepTempo: Learning Temporal Circuit Behaviors of Logic Gates"

# Abstract
With the continuous evolution of integrated circuit technology toward higher complexity and larger scales, modern Electronic Design Automation (EDA) workflows demand sophisticated representation and analysis techniques to achieve optimal circuit design performance, power consumption, and reliability.And-Inverter Graphs (AIGs) serve as a fundamental structure for core EDA tasks, efficiently enabling logic synthesis, equivalence checking, and functional verification to enhance development efficiency.However, the growing complexity and integration density of modern circuits introduce increasingly intricate temporal signal propagation patterns and multi-layered feature interactions, which leads existing works ignoring temporal signal propagation dependencies and facing feature homogenization among deep aggregation in the accurate AIG representation learning.To address these challenges, we propose DeepTempo, a spatio-temporal joint modeling framework that effectively captures temporal signal propagation dependencies and preserves spatial features during deep aggregation to enable accurate AIG representation learning.Specifically, DeepTempo incorporates temporal convolution mechanisms to model sequential signal propagation, to capture the temporal dependencies inherent in circuit behavior.In addition, DeepTempo employs adaptive gating and cross-reconstruction strategies to preserve gate-specific functional distinctiveness during multi-layer feature aggregation, effectively mitigating feature homogenization problems to enhance discriminative modeling and achieve accurate AIG representation.We conduct comprehensive experiments on 9,933 valid circuit samples from multiple EDA benchmark domains covering logic synthesis, circuit testing, and hardware design to evaluate the effectiveness of DeepTempo in AIG representation learning.Experimental results on the Signal Probability Prediction (SPP) task and Truth-Table Distance Prediction (TTDP) task demonstrate that DeepTempo outperforms state-of-the-art methods, achieving improvements of 14.29\% (MAE) and 20.0\% (MSE) on SPP, and achieving 21.35\% (MAE) and 4.55\% (MSE) on TTDP.

# Requirements
Python >= 3.9
PyTorch 2.0.1 (with CUDA 11.8 support)
PyTorch Geometric 2.3.1
torch-scatter 2.1.2
torch-sparse 0.6.18
NumPy 1.22.4
pandas 1.4.2
scikit-learn 1.6.1
tqdm 4.65.0
matplotlib 3.5.1 

# Usage
python train.py --task_type prob --split_file 0.05-0.05-0.9 --layer_num 9
