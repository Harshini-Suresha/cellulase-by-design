"""Design generators: central composite (k = 2-5) and Plackett-Burman (12, 20, 24 runs). numpy/pandas only."""
import itertools
import numpy as np
import pandas as pd

# First rows of the standard cyclic Plackett-Burman designs (Plackett & Burman, 1946).
_PB_ROWS = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def plackett_burman(n):
    """n-run design as an n x (n-1) matrix of -1/+1 (cyclic shifts of the generator plus a row of -1)."""
    if n not in _PB_ROWS:
        raise ValueError(f"Supported run counts: {sorted(_PB_ROWS)}")
    g = np.array([1 if c == "+" else -1 for c in _PB_ROWS[n]])
    rows = [np.roll(g, i) for i in range(n - 1)] + [-np.ones(n - 1, int)]
    return np.array(rows)


def ccd(k, alpha="rotatable", n_center=5, fraction=0):
    """Coded central composite design. fraction p builds a 2^(k-p) factorial using standard generators (k>=5)."""
    if not 2 <= k <= 5:
        raise ValueError("k must be between 2 and 5")
    if k == 5 and fraction == 1:
        fac = np.array(list(itertools.product([-1, 1], repeat=4)))
        fac = np.column_stack([fac, fac.prod(axis=1)])  # E = ABCD, resolution V
    else:
        fac = np.array(list(itertools.product([-1, 1], repeat=k)))
    nf = len(fac)
    a = {"rotatable": nf ** 0.25, "face-centred": 1.0, "spherical": k ** 0.5}[alpha]
    ax = np.zeros((2 * k, k))
    for j in range(k):
        ax[2 * j, j], ax[2 * j + 1, j] = -a, a
    pts = np.vstack([fac, ax, np.zeros((n_center, k))])
    kind = ["Factorial"] * nf + ["Axial"] * (2 * k) + ["Centre"] * n_center
    return pts, kind, a


def build(coded, kind, names, lows, highs, seed=None, std_order=True):
    """Natural-unit run sheet with standard order and (optionally) randomised run order."""
    centre = (np.array(lows) + np.array(highs)) / 2
    half = (np.array(highs) - np.array(lows)) / 2
    nat = centre + coded * half
    df = pd.DataFrame(nat, columns=names)
    df.insert(0, "Type", kind)
    df.insert(0, "Std", np.arange(1, len(df) + 1))
    for j, n in enumerate(names):
        df[f"{n} (coded)"] = np.round(coded[:, j], 4)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "Run", np.arange(1, len(df) + 1))
    df["Response"] = ""
    return df


def properties(coded):
    """Orthogonality / balance diagnostics shown to the user."""
    X = np.asarray(coded, float)
    n, k = X.shape
    G = X.T @ X
    off = np.abs(G - np.diag(np.diag(G))).max()
    return dict(runs=n, factors=k, max_offdiag=float(off), balanced=bool(np.allclose(X.sum(axis=0), 0)))


def full_factorial(k):
    """Two-level full factorial, coded -1/+1 (standard order, first factor varies fastest)."""
    return np.array([[1 if (m >> j) & 1 else -1 for j in range(k)] for m in range(2 ** k)], float)


def box_behnken(k, n_center=3):
    """Box-Behnken design for 3-5 factors: +-1 on every pair of factors, others at 0, plus centre points."""
    if not 3 <= k <= 5:
        raise ValueError("Box-Behnken needs 3-5 factors")
    rows = []
    for a in range(k):
        for b in range(a + 1, k):
            for sa, sb in itertools.product([-1, 1], repeat=2):
                r = np.zeros(k); r[a], r[b] = sa, sb; rows.append(r)
    pts = np.vstack([np.array(rows), np.zeros((n_center, k))])
    return pts, ["Edge"] * len(rows) + ["Centre"] * n_center
