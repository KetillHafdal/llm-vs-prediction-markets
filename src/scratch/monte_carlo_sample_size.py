import numpy as np
from scipy.stats import wilcoxon       # :contentReference[oaicite:2]{index=2}

def power_wilcoxon(n, delta, sd, alpha=0.05, nrep=10000):
    """Estimate power of paired Wilcoxon test by simulation."""
    hits = 0
    for _ in range(nrep):
        diffs = np.random.normal(loc=delta, scale=sd, size=n)
        try:
            stat, p = wilcoxon(diffs)
            if p < alpha:
                hits += 1
        except:
            continue  # skip if test fails due to ties or all-zero diffs
    return hits / nrep

# Example: Low-volume bucket
delta = 0.175    # model - market
sd    = 0.309
alpha = 0.05
nrep  = 10000

# Find the smallest n that reaches 80% power
for n in range(26, 300, 1):
    power = power_wilcoxon(n, delta, sd, alpha, nrep)
    print(f"n={n}: power={power:.3f}")
    if power >= 0.80:
        print(f"\n✅ Target power reached at n = {n}")
        break
#Target power at 10 for low power 0.997 
#Target power reached at n=10 power 0.991 mid 
#Target power reached at n=28 power 0.806

