# cs323_project4

data privacy-project 4 by Hyeoseo Lee

How to reproduce my results:

This project runs using randomly generated input values between 0 and 100. Because of this randomness, the exact numerical results or graphs may not be identical every time, but you will obtain a very similar trend and outcome.

The values of n are fixed as [5, 10, 25, 50, 100].
For each n, the code automatically performs ten runs to compute the average runtime of all four methods — Non-private, Paillier, Shamir, and Differential Privacy (DP).
It also compares each method’s average result with the Non-private output to calculate the average error (accuracy), and then generates two graphs: one showing the runtime comparison among the methods, and another showing their accuracy differences.

To run: `python project4.py`

The resulting graphs which are "average_runtime.png" and "accuracy.png" will be saved automatically.