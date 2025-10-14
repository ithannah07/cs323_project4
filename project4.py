import random
import time
import math
import numpy as np
from phe import paillier
import matplotlib.pyplot as plt

P = 1009

def generate_value(n, min_val = 0, max_val = 100):
    return [random.randint(min_val, max_val) for _ in range(n)]

def computing_time(start, end):
    # print("Computing time: ", end - start, "seconds")
    elasped = end - start
    return float(round(elasped, 10))

def non_private(values):
    start = time.time()
    mean = float(round(np.mean(values), 3))
    end = time.time()
    elasped = computing_time(start, end)
    
    return mean, elasped


def paillier_average(values):
    start = time.time()

    public_key, private_key = paillier.generate_paillier_keypair()

    enc_values = [public_key.encrypt(x) for x in values]

    enc_sum = enc_values[0]
    for i in range(1, len(enc_values)):
        enc_sum += enc_values[i]
    
    dec_sum = private_key.decrypt(enc_sum)

    mean = round((dec_sum / len(values)), 3)

    end = time.time()
    elasped = computing_time(start, end)

    return mean, elasped

def shamir_secret_sharing(values, n):
    start = time.time()
    # t = floor(n/2)
    t = max(2, math.floor(n/2))
    recovered = []

    # 1. Define polynomial f(x) of degree t-1 for each value
    for v in values:
        coeffs = generate_polynomial(v, t)

        # 2. generating n share (i, f(i))
        shares = generate_shares(coeffs, n)
        # print("shares: ", shares)
    
        # 3. randomly select at least t shares (trusted parties
        selected_shares = t_shares(shares, t)
        # print("selected shares: ", selected_shares)

        # 4. reconstructing the secret value
        reconstruction_value = reconstruction(selected_shares)
        # print("reconstruction value: ", reconstruction_value)
        recovered.append(reconstruction_value)

        if reconstruction_value != v:
            print("Original value: ", v, "Reconstructed value: ", reconstruction_value)
            print("Error in reconstruction!")

    # 5. computing the average after recovering all values
    mean = float(round(sum(recovered) / len(recovered), 3))
    end = time.time()
    elasped = computing_time(start, end)

    return mean, elasped

def generate_polynomial(value, t, p = P):
    degree = t - 1
    coeffs = [value % p] + [random.randint(1, p-1) for _ in range(degree)]
    return coeffs

def computing_polynomial(coeffs, x, p = P):
    result = 0
    for i, coeff in enumerate(coeffs):
        result = (result + coeff * pow(x, i, p)) % p
    return result

def generate_shares(coeffs, n, p = P):
    shares = []
    for i in range(1, n + 1):
        result = computing_polynomial(coeffs, i, p)
        shares.append((i, result))
    return shares

def t_shares(shares, t):
    return random.sample(shares, t)

def mod_inverse(num, p = P):
    return pow(num, -1, p)

def reconstruction(selected_shares, p = P):
    """
    selected shares: [(x1, y1), (x2, y2), ..., (xt, yt)]
    f_i(0) = Σ f_i(j) * l_i(j)
    """
    xs = [x for x, _ in selected_shares]
    ys = [y for _, y in selected_shares]

    secret = 0

    for j in range(len(xs)):
        num = 1
        den = 1
        for k in range(len(xs)):
            if k != j:
                num = (num * - xs[k]) % p
                den = (den * (xs[j] - xs[k])) % p

        lj = ( num * mod_inverse(den, p)) % p
        secret = (secret + ys[j] * lj) % p

    return secret

def avg_runtime(function, *args, runs = 10):
    times = []
    for _ in range(runs):
        _, t = function(*args)
        times.append(t)
    return sum(times) / runs

def DP(values, epsilon = 0.1):
    start = time.time()

    min_val = 0
    max_val = 100 # range of values

    sensitivity = (max_val - min_val) / len(values)
    average = np.mean(values)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)

    noise_avg = average + noise

    end = time.time()
    elasped = computing_time(start, end)

    return noise_avg, elasped







def main():
    
    # # just print out the values and their averages
    # for n in [5, 10, 25]:
    #     values = generate_value(n)
    #     avg1, t1 = non_private(values)
    #     avg2, t2 = paillier_average(values)
    #     avg3, t3 = shamir_secret_sharing(values, n)
    #     avg4, n_avg4, t4 = DP(values)
    #     print("Non-private, average: ", avg1, "elapsed time: ", t1)
    #     print("Paillier, average: ", avg2, "elapsed time: ", t2)
    #     print("Shamir, average: ", avg3, "elapsed time: ", t3)
    #     print("DP, average(original): ", avg4, "average(noise): ", n_avg4, "elapsed time: ", t4)
    #     print()
    
    n_values = [5, 10, 25, 50, 100]
    times_non_private = []
    times_paillier = []
    times_shamir = []
    times_dp = []


    errors_paillier = []
    errors_shamir = []
    errors_dp = []

    for n in n_values:
        values = generate_value(n)

        # compute averages
        avg_non_private, _ = non_private(values)
        avg_paillier, _ = paillier_average(values)
        avg_shamir, _ = shamir_secret_sharing(values, n)
        avg_dp, _ = DP(values)

        # compute errors (for accuracy analysis)
        errors_paillier.append(abs(avg_paillier - avg_non_private))
        errors_shamir.append(abs(avg_shamir - avg_non_private))
        errors_dp.append(abs(avg_dp - avg_non_private))

        # conpute average runtimes
        avg_non_private_times = avg_runtime(non_private, generate_value(n))
        avg_paillier_times = avg_runtime(paillier_average, generate_value(n))
        avg_shamir_times = avg_runtime(shamir_secret_sharing, generate_value(n), n)
        avg_dp_times = avg_runtime(DP, generate_value(n))

        times_non_private.append(avg_non_private_times)
        times_paillier.append(avg_paillier_times)
        times_shamir.append(avg_shamir_times)
        times_dp.append(avg_dp_times)

        
    
    # Plotting the results - runtime
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, times_non_private, label = "Non-private")
    plt.plot(n_values, times_paillier, label = "Paillier")
    plt.plot(n_values, times_shamir, label = "Shamir")
    plt.plot(n_values, times_dp, label = "Differential Privacy")
    plt.xticks(n_values)
    plt.xlabel("Number of parties (n)")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Average Runtime vs Number of parties")
    plt.yscale("log") 
    plt.legend()
    plt.grid()
    plt.savefig("average_runtime.png")
    plt.show()
 



    # Plotting the results - accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, errors_paillier, label = "Paillier")
    plt.plot(n_values, errors_shamir, label = "Shamir")
    plt.plot(n_values, errors_dp, label = "Differential Privacy")
    plt.xticks(n_values)
    plt.xlabel("Number of parties (n)")
    plt.ylabel("Absolute Error from Non-private Average")
    plt.title("Accuracy Analysis")
    # plt.yscale("log") 
    plt.legend()
    plt.grid()
    plt.savefig("accuracy.png")
    plt.show()
    




        
    

if __name__ == "__main__":
    main()