# Instructions

We are developing a side channel attack for split learning. The attack in my mind in short is the following. Server will send activations to client and regarding to time differences among updates. Eg. I expect to have bigger time difference among two samples having different labels as the update is less sparse in that case and so on, Or even server can train a side model in side identical to clients setting so that it will now how clients machine behaves under different examples.


# Documents

I am providing a paper on privacy side channels in machine learning named "Privacy Side Channels in Machine Learning Systems". We are trying to implement it to split learning.

We also have a python file where we do the experiments for split learning side channels.

Also we have a experiments directory to save the experiment results.


# Ideas
1. **Activation Reconstruction Attacks**: An adversary reconstructs private data by analyzing intermediate activations exchanged between split learning participants, mitigated by noise injection or differential privacy.  

2. **Gradient Inversion Attacks**: The attacker recovers input data by optimizing a random sample to match observed gradients, countered by gradient obfuscation or differential privacy.  

3. **Message Size-Based Membership Inference**: By analyzing variations in activation sizes, an attacker infers whether specific inputs were used in training, which can be mitigated by padding messages to a uniform size.  

4. **Model Update Timing Analysis**: The attacker measures processing delays to infer properties about input data, which can be mitigated by uniform computation times and randomized batch processing.  

5. **Adversarial Data Poisoning Side-Channels**: A malicious participant introduces subtle modifications to inputs to create detectable patterns in activations or gradients, mitigated by robust aggregation and data sanitization techniques.












