import random
import time
import math
import numpy as np
from phe import paillier
import matplotlib.pyplot as plt

P = 104729  # A large prime number for modulo operations in Shamir's Secret Sharing

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
    start_total = time.time()

    # 1. key generation
    start_phase = time.time()
    public_key, private_key = paillier.generate_paillier_keypair()
    gen_time = time.time() - start_phase

    # 2. encryption
    start_phase = time.time()
    enc_values = [public_key.encrypt(x) for x in values]
    enc_time = time.time() - start_phase

    # 3. aggregation
    start_phase = time.time()
    enc_sum = enc_values[0]
    for i in range(1, len(enc_values)):
        enc_sum += enc_values[i]
    agg_time = time.time() - start_phase

    # 4. decryption
    start_phase = time.time()
    dec_sum = private_key.decrypt(enc_sum)
    dec_time = time.time() - start_phase

    mean = round((dec_sum / len(values)), 3)

    total_time = time.time() - start_total
    timings = {
        "key_generation" : gen_time,
        "encryption" : enc_time,
        "aggregation" : agg_time,
        "decryption" : dec_time,
        "total_time" : total_time
    }

    return mean, timings

def shamir_secret_sharing(values, n):
    start_total = time.time()
    t = max(2, math.floor(n/2))

    # 1. polynomial generation
    start_phase = time.time()

    polys = [generate_polynomial(v, t) for v in values]

    polys_time = time.time() - start_phase

    # 2. share computation
    start_phase = time.time()
    sum_shares = []
    for i in range(1, n+1):
        total = 0
        for coeffs in polys:
            total = (total + computing_polynomial(coeffs, i)) % P
        sum_shares.append((i, total))
    share_time = time.time() - start_phase

    # 3. reconstruction
    start_phase = time.time()
    selected_shares = random.sample(sum_shares, t)
    recovered = reconstruction(selected_shares) # reconstruct the sum of values
    recon_time = time.time() - start_phase

    mean = float(round(recovered / len(values), 3))

    total_time = time.time() - start_total
    timings = {
        "poly_generation" : polys_time,
        "share_computation" : share_time,
        "reconstruction" : recon_time,
        "total_time" : total_time
    }
    return mean, timings

def generate_polynomial(value, t, p = P):
    degree = t - 1
    coeffs = [value % p] + [random.randint(1, p-1) for _ in range(degree)]
    return coeffs

def computing_polynomial(coeffs, x, p = P):
    result = 0
    for i, coeff in enumerate(coeffs):
        result = (result + coeff * pow(x, i, p)) % p
    return result

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

def differential_privacy(values, epsilon = 0.1):
    start_total = time.time()
    min_val = 0
    max_val = 100

    # 1. compute average
    start_phase = time.time()
    average = np.mean(values)
    avg_time = time.time() - start_phase

    # 2. add Laplace noise
    start_phase = time.time()
    sensitivity = (max_val - min_val) / len(values)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    noise_time = time.time() - start_phase

    # 3. noise addition
    start_phase = time.time()
    noise_avg = average + noise
    add_time = time.time() - start_phase

    total_time = time.time() - start_total
    timings = {
        "average_computation" : avg_time,
        "noise_generation" : noise_time,
        "noise_addition" : add_time,
        "total_time" : total_time
    }

    return noise_avg, timings


def avg_runtime(function, *args, runs = 10):
    times = []
    for _ in range(runs):
        _, t = function(*args)
        if isinstance(t, dict):
            times.append(t["total_time"])
        else:
            times.append(t)
    return sum(times) / runs

def avg_phase_runtime(function, *args, runs = 10):
    results = []
    for _ in range(runs):
        _, timings = function(*args)
        results.append(timings)

    keys = results[0].keys()
    avg_dict = {k: sum(d[k] for d in results) / runs for k in keys}
    return avg_dict  # {'phase1': x, 'phase2': y, ...}


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

    phase_paillier = []
    phase_shamir = []
    phase_dp = []

    for n in n_values:
        values = generate_value(n)

        # compute averages
        avg_non_private, _ = non_private(values)
        avg_paillier, _ = paillier_average(values)
        avg_shamir, _ = shamir_secret_sharing(values, n)
        avg_dp, _ = differential_privacy(values)

        # compute errors (for accuracy analysis)
        errors_paillier.append(abs(avg_paillier - avg_non_private))
        errors_shamir.append(abs(avg_shamir - avg_non_private))
        errors_dp.append(abs(avg_dp - avg_non_private))

        # conpute average runtimes
        avg_non_private_times = avg_runtime(non_private, generate_value(n))
        avg_paillier_times = avg_runtime(paillier_average, generate_value(n))
        avg_shamir_times = avg_runtime(shamir_secret_sharing, generate_value(n), n)
        avg_dp_times = avg_runtime(differential_privacy, generate_value(n))

        # runtime
        times_non_private.append(avg_non_private_times)
        times_paillier.append(avg_paillier_times)
        times_shamir.append(avg_shamir_times)
        times_dp.append(avg_dp_times)

        # phase
        phase_paillier.append(avg_phase_runtime(paillier_average, generate_value(n)))
        phase_shamir.append(avg_phase_runtime(shamir_secret_sharing, generate_value(n), n))
        phase_dp.append(avg_phase_runtime(differential_privacy, generate_value(n))) 

        
    
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
    

    # phase runtime for n = 100
    idx = n_values.index(100)

    p_phases = {k: v for k, v in phase_paillier[idx].items() if k != "total_time"}
    s_phases = {k: v for k, v in phase_shamir[idx].items() if k != "total_time"}
    d_phases = {k: v for k, v in phase_dp[idx].items() if k != "total_time"}

    fig, ax = plt.subplots(1, 3, figsize=(14, 5))

    # Paillier
    ax[0].bar(p_phases.keys(), p_phases.values(), color='skyblue')
    ax[0].set_title("Paillier Phases")
    ax[0].set_ylabel("Runtime (seconds, log scale)")
    ax[0].set_yscale("log")
    ax[0].grid(True, which="both", ls="--", lw=0.5)
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Shamir
    ax[1].bar(s_phases.keys(), s_phases.values(), color='orange')
    ax[1].set_title("Shamir Phases")
    ax[1].set_yscale("log")
    ax[1].grid(True, which="both", ls="--", lw=0.5)
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # DP
    ax[2].bar(d_phases.keys(), d_phases.values(), color='lightgreen')
    ax[2].set_title("Differential Privacy Phases")
    ax[2].set_yscale("log")
    ax[2].grid(True, which="both", ls="--", lw=0.5)
    plt.setp(ax[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout(pad=2.0)
    plt.savefig("phase_runtime_comparison.png")
    plt.show()





        
    

if __name__ == "__main__":
    main()