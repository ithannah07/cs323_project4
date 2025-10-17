# cs323_project4

data privacy-project 4 by Hyeoseo Lee

How to reproduce my results:

This project runs using randomly generated input values between 0 and 100. Because of this randomness, the exact numerical results or graphs may not be identical every time, but you will obtain a very similar trend and outcome.

The values of n are fixed as [5, 10, 25, 50, 100].
For each n, the code automatically performs ten runs to compute the average runtime of all four methods — Non-private, Paillier, Shamir, and Differential Privacy.
It also compares each method’s average result with the Non-private output to calculate the average error (accuracy), and then generates two graphs: one showing the runtime comparison among the methods, and another showing their accuracy differences.

In addition, a phase-level runtime graph is included to visualize how much time each method spends in its internal phases.
This third graph (phase_runtime_comparison.png) breaks down the total runtime into individual steps — for example, Paillier’s key generation, encryption, aggregation, and decryption; Shamir’s polynomial generation, share computation, and reconstruction; and differential privacy’s mean calculation, noise generation, and addition.
This allows a clear comparison of which phases dominate the overall runtime for each privacy-preserving method.

To run: `python project4.py`

The resulting graphs which are "average_runtime.png," "accuracy.png," and "phase_runtime_comparison.png" will be saved automatically.