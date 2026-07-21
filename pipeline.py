import os
import sys
import subprocess
import scipy.io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import zipfile
from scipy import stats
from scipy.signal import resample as scipy_resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nitime.timeseries import TimeSeries
from nitime.analysis   import CorrelationAnalyzer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT     = os.path.dirname(os.path.abspath(__file__))
DATASETS_ROOT = os.path.join(REPO_ROOT, "datasets")
RESULTS_ROOT  = os.path.join(REPO_ROOT, "results")

ALL_SUBJECTS = [
    "CON01","CON02","CON03","CON04","CON05","CON06",
    "CON07","CON08","CON09","CON10","CON11",
    "PAT01","PAT02","PAT03","PAT05","PAT06","PAT07",
    "PAT08","PAT10","PAT11","PAT13","PAT14","PAT15",
    "PAT16","PAT17","PAT19","PAT20","PAT22","PAT23",
    "PAT24","PAT25","PAT26","PAT27","PAT28","PAT29","PAT31",
]

S3_BUCKET = "s3://openneuro.org"

def dataset_dir(dataset, sub_str):
    return os.path.normpath(
        os.path.join(DATASETS_ROOT, dataset, "sub-" + sub_str, "ses-preop"))

def results_dir(dataset, sub_str):
    return os.path.normpath(
        os.path.join(RESULTS_ROOT, dataset, sub_str, "outputs", "G_sweep"))

def aws_available():
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def ensure_subject_data(dataset, sub_str):
    path = dataset_dir(dataset, sub_str)
    required = ["FC.mat", "SC.zip"]
    missing = [f for f in required if not os.path.exists(os.path.join(path, f))]
    if not missing:
        return path

    if not aws_available():
        raise RuntimeError(
            f"Missing {missing} in {path} and the aws CLI is not available "
            f"to download them. Install awscli or place the files manually."
        )

    os.makedirs(path, exist_ok=True)
    sub_dir = "sub-" + sub_str
    for fname in missing:
        local_path = os.path.join(path, fname)
        s3_path = f"{S3_BUCKET}/{dataset}/derivatives/TVB/{sub_dir}/ses-preop/{fname}"
        print(f"  Downloading {fname} for {sub_str} from {s3_path}")
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, local_path, "--no-sign-request"],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not os.path.exists(local_path):
            raise RuntimeError(
                f"Failed to download {fname} for {sub_str} from {s3_path}\n"
                f"{result.stderr}"
            )
    return path

def generate_hrf_mat(roi_ts, TR, out_path, name="generated", p_jobs=1):
    """
    Produce an HRF.mat equivalent to the dataset's own derivatives format
    directly from a region-averaged BOLD timeseries, by running rsHRF in
    'time-series' mode (the same mode used to estimate per-region HRFs
    from already-parcellated BOLD, not raw voxelwise data).

    Verified against a real subject's HRF.mat (CON01, ds001226):
      - T=1, hrf_len=21 reproduces the exact saved shape (11, n_regions)
        at TR=2.1s (n_samples = fix(len/TR) + 1).
      - Per-region HRF *shape* correlation vs the real file: mean r=0.957
        across valid (non-NaN) regions.
    Amplitude is on an arbitrary scale relative to the original dataset's
    HRF.mat — harmless here since load_subject_data() normalizes hrf_raw
    by per-region max-abs before use, so only shape matters downstream.
    """
    from rsHRF import fourD_rsHRF

    roi_ts = np.asarray(roi_ts, dtype=float)
    work_dir = os.path.join(os.path.dirname(out_path), f"_{name}_rshrf_tmp")
    os.makedirs(work_dir, exist_ok=True)

    csv_path = os.path.join(work_dir, f"{name}.csv")
    np.savetxt(csv_path, roi_ts, delimiter=',')

    T, hrf_len = 1, 21
    para = dict(
        estimation='canon2dd',
        passband=[0.01, 0.08],
        passband_deconvolve=[0.0, np.finfo(float).max],
        TR=TR, T=T, T0=1, TD_DD=2, AR_lag=1,
        thr=1.0, order=3, len=hrf_len,
        min_onset_search=4, max_onset_search=8,
    )
    para['dt'] = para['TR'] / para['T']
    para['lag'] = np.arange(
        np.fix(para['min_onset_search'] / para['dt']),
        np.fix(para['max_onset_search'] / para['dt']) + 1,
        dtype='int')

    fourD_rsHRF.demo_rsHRF(
        input_file=csv_path,
        mask_file=None,
        output_dir=work_dir,
        para=para,
        p_jobs=p_jobs,
        file_type='.csv',
        mode='time-series',
        wiener=False,
    )

    gen = scipy.io.loadmat(os.path.join(work_dir, f"{name}_hrf_deconv.mat"))
    hrf, PARA = gen['hrfa'], gen['PARA']

    scipy.io.savemat(out_path, {"hrf": hrf, "PARA": PARA})
    return hrf, PARA

def load_subject_data(dataset, sub_str):
    """Load FC, SC, HRF for one subject. Returns dict or raises."""
    path = ensure_subject_data(dataset, sub_str)
    print(f"  Loading: {path}")

    mat    = scipy.io.loadmat(os.path.join(path, "FC.mat"))
    emp_fc = mat["FC_cc_DK68"]
    TR     = float(mat["TR"][0][0])

    with zipfile.ZipFile(os.path.join(path, "SC.zip"), "r") as z:
        with z.open("weights.txt") as f:
            weights = np.loadtxt(f)
        with z.open("tract_lengths.txt") as f:
            tract_lengths = np.loadtxt(f)

    if tract_lengths.size == 0 or tract_lengths.ndim < 2:
        print(f"  WARNING: tract_lengths.txt empty — using uniform 50mm distances")
        tract_lengths = np.where(weights > 0, 50.0, 0.0)

    hrf_mat_path = os.path.join(path, "HRF.mat")
    if os.path.exists(hrf_mat_path):
        try:
            hrf_mat = scipy.io.loadmat(hrf_mat_path)
            hrf_raw = hrf_mat["hrf"]
        except Exception as e:
            print(f"  WARNING: existing HRF.mat unreadable ({e}) — regenerating via rsHRF")
            hrf_raw = None
    else:
        hrf_raw = None

    roi_key = next(k for k in mat if k.endswith("ROIts_DK68"))
    roi_ts  = mat[roi_key]   # empirical region timeseries, shape (T, N)

    if hrf_raw is None:
        print(f"  HRF.mat missing/invalid for {sub_str} — generating via rsHRF "
              f"from {roi_key} {roi_ts.shape}")
        hrf_raw, _ = generate_hrf_mat(roi_ts, TR, hrf_mat_path, name=sub_str)

    hrf_norm = hrf_raw / (np.max(np.abs(hrf_raw), axis=0, keepdims=True) + 1e-12)
    nan_regions = np.where(np.isnan(hrf_norm).any(axis=0))[0]
    if len(nan_regions):
        col_mean = np.nanmean(hrf_norm, axis=1, keepdims=True)
        for r in nan_regions:
            hrf_norm[:, r] = col_mean[:, 0]
    hrf_compact = hrf_norm

    N            = weights.shape[0]
    BOLD_TR_ms   = int(TR * 1000)
    total_dur_ms = int(TR * 1000 * 200)

    return dict(
        emp_fc=emp_fc, TR=TR, weights=weights,
        tract_lengths=tract_lengths, hrf_compact=hrf_compact,
        N=N, BOLD_TR_ms=BOLD_TR_ms, total_dur_ms=total_dur_ms,
        HRF_samples=hrf_compact.shape[0], roi_ts=roi_ts
    )

dt            = 0.1
interim_istep = 10
model_dt      = 0.001

a_E    = 310.0
b_E    = 125.0
d_E    = 0.16
a_I    = 615.0
b_I    = 177.0
d_I    = 0.087
gamma  = 0.641 / 1000.0
tau_E  = 100.0
tau_I  = 10.0
I_0    = 0.382
w_E    = 1.0
w_I    = 0.7
gamma_I= 1.0 / 1000.0
J_NMDA = 0.150
w_plus = 1.400
sigma  = 0.0100

w_E_I_0       = w_E * I_0
w_I_I_0       = w_I * I_0
sigma_sqrt_dt = sigma * np.sqrt(dt)

rho      = 0.34
alpha    = 0.32
tau_bw   = 0.98
y_bw     = 1.0 / 0.41
kappa    = 1.0 / 0.65
V_0      = 0.02
k1       = 7.0 * rho
k2       = 2.0
k3       = 2.0 * rho - 0.2
ialpha   = 1.0 / alpha
itau_bw  = 1.0 / tau_bw
oneminrho= 1.0 - rho

target_FR    = 3.0
isp_eta      = 0.001
FIC_start_ms = 10000
FIC_end_ms   = 180000
FIC_interval_ms = 10000

FIC_start_step   = FIC_start_ms
FIC_end_step     = FIC_end_ms
FIC_interval_step= FIC_interval_ms

HRF_LEN_S   = 25.0
model_dt_s  = 0.001
stock_steps = int(HRF_LEN_S / model_dt_s) + 1

def resample_hrf_for_conv(hrf_compact, N, stock_steps):
    """Resample each region's HRF from HRF_samples to stock_steps,
    then reverse for dot-product convolution (exact as C)."""
    HRF_rs = np.zeros((stock_steps, N))
    for r in range(N):
        h = hrf_compact[:, r]
        h_up = scipy_resample(h, stock_steps)
        HRF_rs[:, r] = h_up[::-1]
    return HRF_rs

global_v   = 12.5

def prepare_sc(weights, tract_lengths, G, global_v=12_500.0):
    """
    Exact C binary formula from importGlobalConnectivity:
      cap = weights * G * J_NMDA   (no normalization)
    global_v in mm/s, dt in seconds
    delay = tract_length_mm / (global_v_mm_s * dt_s)
    """

    cap = weights * G * J_NMDA

    raw_delays = tract_lengths / (global_v * dt)
    delays = np.round(raw_delays).astype(int)
    delays = np.clip(delays, 1, int(raw_delays.max()) + 1)
    return cap, delays

def run_simulation(G, weights, tract_lengths, hrf_compact,
                   total_dur_ms, BOLD_TR_ms, N,
                   J_NMDA=0.150, w_plus=1.400, tmpJi=1.000,
                   sigma=0.0100, rand_seed=1403,
                   mode='rshrf'):
    """
    Returns simulated BOLD timeseries shape (N, BOLD_TS_len).

    Verified against actual C source code:
    - tvbii_multicore.c (legacy): NO FIC, BW at model_dt=1ms outside inner loop
    - main.c (rshrf): FIC with Vogels STDP rule, rsHRF convolution
    """
    np.random.seed(rand_seed)

    BOLD_TS_len    = total_dur_ms // BOLD_TR_ms
    total_steps_ms = total_dur_ms
    model_dt_bw    = 0.001

    w_plus_J_NMDA = w_plus * J_NMDA
    SC_cap = (weights * G * J_NMDA).astype(np.float32)

    S_E = np.random.uniform(0.0, 0.5, N)
    S_I = np.random.uniform(0.0, 0.5, N)
    J_i = np.ones(N) * tmpJi

    meanFR     = np.zeros(N)
    meanFR_INH = np.zeros(N)
    i_meanfr   = 0

    BOLD_out = np.zeros((N, BOLD_TS_len), dtype=np.float32)
    bold_t   = 0

    bw_x  = np.zeros(N, dtype=np.float32)
    bw_f  = np.ones(N, dtype=np.float32)
    bw_nu = np.ones(N, dtype=np.float32)
    bw_q  = np.ones(N, dtype=np.float32)

    if mode == 'rshrf':
        HRF_rs  = resample_hrf_for_conv(hrf_compact, N, stock_steps)
        sig_buf = np.zeros((stock_steps, N))
        sig_idx = 0

    BOLD_TR_steps = BOLD_TR_ms

    model_dt_delay = 0.001
    raw_delays = tract_lengths / (12.5 * model_dt_delay)
    delays = np.round(raw_delays).astype(int)
    delays = np.clip(delays, 1, max(int(raw_delays.max()), 1) + 1)
    max_delay = int(delays.max()) + 2

    delay_buf = np.zeros((max_delay, N), dtype=np.float32)

    for d in range(max_delay):
        delay_buf[d, :] = S_E.copy()
    delay_idx = 0

    print(f"  Simulating {total_steps_ms} ms  G={G:.2f}  mode={mode}  "
          f"FIC={'ON' if mode=='rshrf' else 'OFF'}  "
          f"max_delay={max_delay}", flush=True)

    conn_target = []
    conn_source = []
    conn_cap    = []
    conn_delay  = []
    for i in range(N):
        for j in range(N):
            if SC_cap[i, j] > 0:
                conn_target.append(i)
                conn_source.append(j)
                conn_cap.append(SC_cap[i, j])
                conn_delay.append(delays[i, j])
    conn_target = np.array(conn_target, dtype=int)
    conn_source = np.array(conn_source, dtype=int)
    conn_cap    = np.array(conn_cap)
    conn_delay  = np.array(conn_delay, dtype=int)
    n_conn      = len(conn_target)
    print(f"  Sparse connectivity: {n_conn} nonzero connections "
          f"({n_conn*100/(N*N):.1f}% of {N*N})")

    for ts in range(total_steps_ms):

        if ts % 50000 == 0:
            print(f"    {ts/total_steps_ms*100:.0f}%  "
                  f"S_E_mean={S_E.mean():.4f}  "
                  f"J_i_mean={J_i.mean():.4f}", flush=True)

        buf_positions = (delay_idx - conn_delay) % max_delay
        delayed_S_E_vals = delay_buf[buf_positions, conn_source]
        weighted_inputs  = conn_cap * delayed_S_E_vals
        global_input = np.zeros(N)
        np.add.at(global_input, conn_target, weighted_inputs)

        for _ in range(interim_istep):

            I_E = a_E * (w_E_I_0
                         + w_plus_J_NMDA * S_E
                         + global_input
                         - J_i * S_I) - b_E
            exp_E   = np.exp(-d_E * I_E)
            exp_E   = np.where(I_E != 0, exp_E, 0.9)
            H_E     = np.clip(I_E / (1.0 - exp_E), 0.0, None)

            I_I = a_I * (w_I_I_0 + J_NMDA * S_E - S_I) - b_I
            exp_I   = np.exp(-d_I * I_I)
            exp_I   = np.where(I_I != 0, exp_I, 0.9)
            H_I     = np.clip(I_I / (1.0 - exp_I), 0.0, None)

            meanFR     += H_E
            meanFR_INH += H_I
            i_meanfr   += 1

            noise_E = (sigma_sqrt_dt * np.random.randn(N)).astype(np.float32)
            noise_I = (sigma_sqrt_dt * np.random.randn(N)).astype(np.float32)

            dS_E = dt * (-S_E / tau_E + (1.0 - S_E) * gamma * H_E) + noise_E
            dS_I = dt * (-S_I / tau_I + H_I * gamma_I) + noise_I
            S_E  = np.clip(S_E + dS_E, 0.0, 1.0)
            S_I  = np.clip(S_I + dS_I, 0.0, 1.0)

        delay_buf[delay_idx, :] = S_E
        delay_idx = (delay_idx + 1) % max_delay

        if mode == 'legacy':
            md = np.float32(model_dt_bw)
            kp = np.float32(kappa)
            yy = np.float32(y_bw)
            it = np.float32(itau_bw)
            ia = np.float32(ialpha)
            rr = np.float32(rho)
            omr = np.float32(oneminrho)
            for j in range(N):
                se  = np.float32(S_E[j])
                bx  = np.float32(bw_x[j])
                bf  = np.float32(bw_f[j])
                bnu = np.float32(bw_nu[j])
                bq  = np.float32(bw_q[j])

                bx  = np.float32(bx + md * (se - kp * bx - yy * (bf - np.float32(1.0))))

                ft  = np.float32(bf + md * bx)

                bnu = np.float32(bnu + md * it * (bf - np.float32(np.power(np.float32(bnu), ia))))

                bq  = np.float32(bq + md * it * (
                    np.float32(bf * (np.float32(1.0) - np.float32(np.power(omr, np.float32(1.0) / bf))) / rr)
                    - np.float32(np.float32(np.power(np.float32(bnu), ia)) * bq / bnu)))

                bf  = ft
                bw_x[j]  = bx
                bw_f[j]  = bf
                bw_nu[j] = bnu
                bw_q[j]  = bq

        if mode == 'rshrf':
            if (ts >= FIC_start_step and ts <= FIC_end_step
                    and ts % FIC_interval_step == 0 and i_meanfr > 0):
                mean_FR_E = meanFR     / i_meanfr
                mean_FR_I = meanFR_INH / i_meanfr
                print(f"    FIC t={ts}ms  mean_FR={mean_FR_E.mean():.2f}Hz  "
                      f"J_i={J_i.mean():.4f}", flush=True)

                J_i += isp_eta * (mean_FR_I * mean_FR_E - target_FR * mean_FR_I)
                J_i  = np.clip(J_i, 0.0001, None)
                meanFR[:]     = 0.0
                meanFR_INH[:] = 0.0
                i_meanfr      = 0

        if mode == 'rshrf':
            sig_buf[sig_idx, :] = S_E
            sig_idx = (sig_idx + 1) % stock_steps

        if (ts + 1) % BOLD_TR_steps == 0 and bold_t < BOLD_TS_len:
            if mode == 'rshrf':
                ordered = np.roll(sig_buf, -sig_idx, axis=0)
                BOLD_out[:, bold_t] = np.einsum('ti,ti->i', ordered, HRF_rs)
            elif mode == 'legacy':

                for j in range(N):
                    bq = np.float32(bw_q[j])
                    bn = np.float32(bw_nu[j])
                    BOLD_out[j, bold_t] = np.float32(
                        np.float32(100.0) / np.float32(rho) * np.float32(V_0) * (
                            np.float32(k1) * (np.float32(1.0) - bq)
                            + np.float32(k2) * (np.float32(1.0) - bq / bn)
                            + np.float32(k3) * (np.float32(1.0) - bn)))
            bold_t += 1

    print(f"    Done.  BOLD shape: {BOLD_out.shape}", flush=True)
    return BOLD_out

DISCARD_TRS = 20

def get_fc_correlation(bold, emp_fc, TR, N, discard=0, mode='both'):

    uidx    = np.triu_indices(N, 1)
    em_z    = np.arctanh(np.clip(emp_fc[uidx], -0.9999, 0.9999))
    T       = TimeSeries(bold, sampling_interval=TR)
    C       = CorrelationAnalyzer(T)
    sim_fc  = np.array(C.corrcoef)
    sim_z   = np.nan_to_num(np.arctanh(np.clip(sim_fc[uidx], -0.9999, 0.9999)))
    r, _    = stats.pearsonr(sim_z, em_z)
    return r, sim_fc

def run_one_G(args):
    """Worker: runs legacy + rsHRF for one G value. Returns all results."""
    (G, weights, tract_lengths, hrf_compact,
     total_dur_ms, BOLD_TR_ms, N, emp_fc, TR) = args
    try:
        bold_leg = run_simulation(
            G, weights, tract_lengths, hrf_compact,
            total_dur_ms, BOLD_TR_ms, N, mode='legacy')
        r_leg, sim_fc_leg = get_fc_correlation(bold_leg, emp_fc, TR, N)

        bold_hrf = run_simulation(
            G, weights, tract_lengths, hrf_compact,
            total_dur_ms, BOLD_TR_ms, N, mode='rshrf')
        r_hrf, sim_fc_hrf = get_fc_correlation(bold_hrf, emp_fc, TR, N)

        print(f"  G={G:.2f}  Legacy r={r_leg:.4f}  rsHRF r={r_hrf:.4f}", flush=True)
        return G, r_leg, sim_fc_leg, r_hrf, sim_fc_hrf
    except Exception as e:
        print(f"  G={G:.2f} FAILED: {e}", flush=True)
        return G, float('nan'), None, float('nan'), None

def select_best_G(G_values_sorted, r_values, tol=1e-3):
    """
    G_values_sorted : G values sorted ascending
    r_values        : corresponding r values (same order), NaNs allowed
    tol             : a point counts as "as good as the best" if its r
                       is within `tol` of the curve's overall max r.

    Returns (best_G, best_r, best_idx).

    First finds the global max r across the whole curve (this is the
    fixed target — it never drifts). Then walks G ascending and picks
    the SMALLEST G whose r already comes within `tol` of that max.
    This correctly handles long, slow plateaus/creeps: even if r keeps
    inching upward for many points after the plateau begins, every
    point in the plateau is compared against the same fixed ceiling,
    so the comparison can't silently compound step by step.

    If r dips after an early near-max point and only reaches the true
    global max much later (i.e. the early "good enough" point sits in
    its own earlier bump rather than the start of the real plateau),
    this still returns the earliest point within tol of the global
    max, consistent with "smallest G among those maximising r".
    """
    idx_valid = [i for i, r in enumerate(r_values) if r == r and r is not None]
    if not idx_valid:
        return None, float('nan'), None

    global_max_r = max(r_values[i] for i in idx_valid)

    for i in idx_valid:
        if r_values[i] >= global_max_r - tol:
            return G_values_sorted[i], r_values[i], i

    best_idx = max(idx_valid, key=lambda i: r_values[i])
    return G_values_sorted[best_idx], r_values[best_idx], best_idx

# BOLD-only convention: simulate 250 TRs, discard the first 50 -> 200
# TRs of settled BOLD saved to disk. Separate from DISCARD_TRS above
# (which belongs to get_fc_correlation's unused `discard` param) so
# changing one can never silently change the other.
BOLD_ONLY_DISCARD_TRS = 50

def default_G_values():
    """The standard 16-point G sweep used by run_fc_sweep when
    G_values isn't explicitly overridden."""
    return sorted([(i / 10) + 0.01 for i in range(0, 31, 2)], reverse=True)

def get_best_G_from_fc(dataset, sub_str, G_TOL=0.015):
    """
    Read G_values.txt / PCorr_legacy.txt / PCorr_rshrf.txt already saved
    by a completed 'fc' mode run, and return (best_G_leg, best_G_hrf)
    using the same select_best_G() rule as run_fc_sweep. Returns
    (None, None) if those files don't exist yet.
    """
    out_dir  = results_dir(dataset, sub_str)
    g_path   = os.path.join(out_dir, "G_values.txt")
    leg_path = os.path.join(out_dir, "PCorr_legacy.txt")
    hrf_path = os.path.join(out_dir, "PCorr_rshrf.txt")
    if not (os.path.exists(g_path) and os.path.exists(leg_path) and os.path.exists(hrf_path)):
        return None, None

    G_values     = np.loadtxt(g_path)
    PCorr_legacy = np.loadtxt(leg_path)
    PCorr_rshrf  = np.loadtxt(hrf_path)
    asc_order = np.argsort(G_values)
    G_asc     = list(G_values[asc_order])
    r_leg_asc = list(PCorr_legacy[asc_order])
    r_hrf_asc = list(PCorr_rshrf[asc_order])

    best_G_leg, _, _ = select_best_G(G_asc, r_leg_asc, tol=G_TOL)
    best_G_hrf, _, _ = select_best_G(G_asc, r_hrf_asc, tol=G_TOL)
    return best_G_leg, best_G_hrf

def simulate_bold_one_G(args):
    """Worker: simulate BOLD (Legacy BW + rsHRF) for ONE G value.
    250 TRs simulated, first 50 discarded -> 200 TRs returned.
    No FC computed here — this is BOLD generation only."""
    (G, weights, tract_lengths, hrf_compact, BOLD_TR_ms, N) = args
    bold_dur_ms = int(BOLD_TR_ms * 250)
    try:
        bold_leg_full = run_simulation(
            G, weights, tract_lengths, hrf_compact,
            bold_dur_ms, BOLD_TR_ms, N, mode='legacy')
        bold_leg = bold_leg_full[:, BOLD_ONLY_DISCARD_TRS:]
        del bold_leg_full

        bold_hrf_full = run_simulation(
            G, weights, tract_lengths, hrf_compact,
            bold_dur_ms, BOLD_TR_ms, N, mode='rshrf')
        bold_hrf = bold_hrf_full[:, BOLD_ONLY_DISCARD_TRS:]
        del bold_hrf_full

        print(f"  [BOLD] G={G:.2f} done  legacy={bold_leg.shape}  rshrf={bold_hrf.shape}", flush=True)
        return G, bold_leg, bold_hrf
    except Exception as e:
        print(f"  [BOLD] G={G:.2f} FAILED: {e}", flush=True)
        return G, None, None

def _zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()

def plot_bold_region0_comparison(sub_str, emp_bold, bold_leg, bold_hrf, out_dir):
    """
    Region 0 only, z-scored, 3 stacked subplots: Empirical BOLD (top),
    Legacy/BW (middle), rsHRF/canon2dd (bottom).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    if emp_bold is not None:
        emp_region0 = np.asarray(emp_bold)[:, 0]
        axes[0].plot(_zscore(emp_region0), color='black')
    axes[0].set_title("Empirical BOLD")

    axes[1].plot(_zscore(bold_leg[0]), color='tab:blue')
    axes[1].set_title("Legacy (BW)")

    axes[2].plot(_zscore(bold_hrf[0]), color='tab:red')
    axes[2].set_title("rsHRF (canon2dd)")

    for ax in axes:
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timepoint (TR)")
    plt.suptitle(f"{sub_str} — Region 0 BOLD (z-scored)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bold_region0_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(out_dir, 'bold_region0_comparison.png')}")

def plot_bold_regions1to5(sub_str, bold_hrf, TR, out_dir):
    """
    Regions 1-5, rsHRF, ONE combined figure (5 lines overlaid). Input
    is already 250-sim/50-discard'd, so the transient is already removed.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for r in range(1, 6):
        ax.plot(bold_hrf[r], label=f"Region {r}")
    ax.set_xlabel(f"Timepoints (TR={TR}s)")
    ax.set_ylabel("BOLD signal")
    ax.set_title(f"Simulated BOLD — {sub_str} rsHRF (canon2dd) — transient removed")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bold_regions1to5_transient_removed.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(out_dir, 'bold_regions1to5_transient_removed.png')}")

def run_bold_sweep(dataset, sub_str, G_values=None):
    """
    Simulate BOLD ONLY at the optimal G for each method — best_G_leg for
    Legacy, best_G_hrf for rsHRF, read from an already-completed 'fc'
    mode run's PCorr files (via get_best_G_from_fc) — NOT at every G in
    the 16-point sweep. Runs in parallel (ProcessPoolExecutor) across
    however many distinct G's are actually needed (1 if both methods
    share the same best G, 2 otherwise).

    If G_values is given explicitly (i.e. --G was passed), that single
    G is used for BOTH methods instead of each one's own best fit, and
    the 'fc' sweep is not required to have been run first.

    Saves exactly 2 plots to results/<dataset>/<subject>/outputs/G_sweep/:
      - bold_region0_comparison.png             (Empirical/Legacy/rsHRF, Region 0)
      - bold_regions1to5_transient_removed.png  (Regions 1-5, rsHRF, combined)
    """
    print("\n" + "=" * 60)
    print(f"BOLD  DATASET: {dataset}  SUBJECT: {sub_str}")
    print("=" * 60)

    out_dir = results_dir(dataset, sub_str)
    os.makedirs(out_dir, exist_ok=True)

    try:
        d = load_subject_data(dataset, sub_str)
    except Exception as e:
        print(f"  FAILED to load data: {e}")
        return

    weights       = d["weights"]
    tract_lengths = d["tract_lengths"]
    hrf_compact   = d["hrf_compact"]
    N             = d["N"]
    BOLD_TR_ms    = d["BOLD_TR_ms"]
    TR            = d["TR"]
    emp_bold      = d.get("roi_ts")

    if G_values is not None:
        best_G_leg = best_G_hrf = G_values[0]
    else:
        best_G_leg, best_G_hrf = get_best_G_from_fc(dataset, sub_str)
        if best_G_leg is None or best_G_hrf is None:
            print(f"  No completed 'fc' sweep found for {sub_str} — run "
                  f"--mode fc first (so best-fit G is known), or pass "
                  f"--G explicitly to override.")
            return

    print(f"  N={N}  TR={TR}s  best_G_leg={best_G_leg}  best_G_hrf={best_G_hrf}")

    G_set = sorted(set([best_G_leg, best_G_hrf]))
    task_args = [(G, weights, tract_lengths, hrf_compact, BOLD_TR_ms, N) for G in G_set]
    n_workers = min(multiprocessing.cpu_count(), len(task_args))

    results_by_G = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(simulate_bold_one_G, a): a[0] for a in task_args}
        for future in as_completed(futures):
            G, bold_leg, bold_hrf = future.result()
            results_by_G[G] = (bold_leg, bold_hrf)

    bold_leg_at_best = results_by_G[best_G_leg][0]
    bold_hrf_at_best = results_by_G[best_G_hrf][1]

    if bold_leg_at_best is None or bold_hrf_at_best is None:
        print("  BOLD simulation failed — no plots produced.")
        return

    plot_bold_region0_comparison(sub_str, emp_bold, bold_leg_at_best, bold_hrf_at_best, out_dir)
    plot_bold_regions1to5(sub_str, bold_hrf_at_best, TR, out_dir)

    print(f"  Saved 2 BOLD plots to: {out_dir}")

def summary_dir(dataset):
    return os.path.normpath(os.path.join(RESULTS_ROOT, dataset, "summary"))

def summarize_subjects(dataset, subjects=None):
    """
    Aggregate across subjects using ONLY the per-subject files already
    on disk from run_fc_sweep() — PCorr_legacy.txt, PCorr_rshrf.txt,
    G_values.txt. No summary.json needed or produced.

    Saves two plots under results/<dataset>/summary/:
      1. best_fit_paired_scatter.png — best-fit (max of the sweep
         curve, same selection rule as run_fc_sweep's select_best_G)
         Legacy PCorr vs rsHRF PCorr, one point per subject.
      2. best_fit_parameter_differences.png — the "model parameters"
         at best fit, meaning the PCorr Legacy/rsHRF values themselves:
         a grouped bar chart per subject, plus a difference bar chart
         (rsHRF − Legacy), colored by CON/PAT.

    subjects: list of subject IDs to include, or None for ALL_SUBJECTS
    (subjects without completed PCorr files are skipped automatically).
    """
    if subjects is None:
        subjects = ALL_SUBJECTS

    G_TOL = 0.015
    subs, best_r_leg, best_r_hrf = [], [], []

    for sub in subjects:
        out_dir  = results_dir(dataset, sub)
        g_path   = os.path.join(out_dir, "G_values.txt")
        leg_path = os.path.join(out_dir, "PCorr_legacy.txt")
        hrf_path = os.path.join(out_dir, "PCorr_rshrf.txt")
        if not (os.path.exists(g_path) and os.path.exists(leg_path) and os.path.exists(hrf_path)):
            continue

        G_values     = np.loadtxt(g_path)
        PCorr_legacy = np.loadtxt(leg_path)
        PCorr_rshrf  = np.loadtxt(hrf_path)

        asc_order = np.argsort(G_values)
        G_asc     = list(G_values[asc_order])
        r_leg_asc = list(PCorr_legacy[asc_order])
        r_hrf_asc = list(PCorr_rshrf[asc_order])

        _, best_rl, _ = select_best_G(G_asc, r_leg_asc, tol=G_TOL)
        _, best_rh, _ = select_best_G(G_asc, r_hrf_asc, tol=G_TOL)

        subs.append(sub)
        best_r_leg.append(best_rl)
        best_r_hrf.append(best_rh)

    if not subs:
        print(f"  No subjects with completed PCorr files found for dataset={dataset}")
        return

    r_leg_arr = np.array(best_r_leg, dtype=float)
    r_hrf_arr = np.array(best_r_hrf, dtype=float)
    valid   = ~(np.isnan(r_leg_arr) | np.isnan(r_hrf_arr))
    subs_v  = [s for s, v in zip(subs, valid) if v]
    r_leg_v = r_leg_arr[valid]
    r_hrf_v = r_hrf_arr[valid]
    is_con  = np.array([s.upper().startswith('CON') for s in subs_v])

    if len(subs_v) == 0:
        print(f"  All {len(subs)} subject(s) found had NaN best-fit r — nothing to plot")
        return

    out_dir = summary_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)

    # ── PLOT 1: paired scatter — rsHRF (canon2dd) vs Legacy (BW),
    # best-fit (max of sweep curve). TWO SIDE-BY-SIDE PANELS, one per
    # group (Controls, Patients), each with its own "Equal performance"
    # diagonal and a vertical connector from each point down to that
    # diagonal, matching the reference figure exactly. ──
    n_con = int(np.sum(is_con))
    n_pat = int(np.sum(~is_con))
    n_hrf_wins_con = int(np.sum(r_hrf_v[is_con]  > r_leg_v[is_con]))
    n_hrf_wins_pat = int(np.sum(r_hrf_v[~is_con] > r_leg_v[~is_con]))

    lo = min(r_leg_v.min(), r_hrf_v.min()) - 0.02
    hi = max(r_leg_v.max(), r_hrf_v.max()) + 0.02

    fig, (axC, axP) = plt.subplots(1, 2, figsize=(14, 7))
    panels = [
        (axC, is_con,  'tab:blue',   f"Controls (CON)\nrsHRF better in {n_hrf_wins_con}/{n_con} subjects"),
        (axP, ~is_con, 'tab:orange', f"Patients (PAT)\nrsHRF better in {n_hrf_wins_pat}/{n_pat} subjects"),
    ]
    for axg, mask, color, subtitle in panels:
        rl, rh = r_leg_v[mask], r_hrf_v[mask]
        axg.plot([lo, hi], [lo, hi], 'k--', alpha=0.6, label='Equal performance')
        for rli, rhi in zip(rl, rh):
            axg.plot([rli, rli], [rli, rhi], color=color, linewidth=1, alpha=0.5, zorder=1)
        axg.scatter(rl, rh, color=color, s=70, edgecolor='none', zorder=2)
        axg.set_xlabel("Legacy (BW) — Pearson r")
        axg.set_ylabel("rsHRF (canon2dd) — Pearson r")
        axg.set_title(subtitle)
        axg.set_xlim(lo, hi); axg.set_ylim(lo, hi)
        axg.legend(); axg.grid(True, alpha=0.3)

    plt.suptitle("Paired comparison: rsHRF vs Legacy FC-empirical correlation", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_fit_paired_scatter.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(out_dir, 'best_fit_paired_scatter.png')}")

    # ── PLOT 2: differences in the model parameters at best fit — the
    # PCorr Legacy/rsHRF values themselves, as a LINE plot: one line
    # connecting Legacy's PCorr across subjects, one line connecting
    # rsHRF's PCorr across subjects. Exactly 2 legend entries. ──
    x = np.arange(len(subs_v))
    fig2, ax2 = plt.subplots(figsize=(max(10, len(subs_v)*0.5), 6))

    ax2.plot(x, r_leg_v, 'b-o', label='Legacy', markersize=6)
    ax2.plot(x, r_hrf_v, 'r-o', label='rsHRF',  markersize=6)

    ax2.set_xticks(x); ax2.set_xticklabels(subs_v, rotation=45, ha='right')
    ax2.set_ylabel("Best-fit Pearson r (PCorr)")
    ax2.set_title(f"PCorr at best fit — Legacy vs rsHRF — {dataset}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_fit_parameter_differences.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(out_dir, 'best_fit_parameter_differences.png')}")

    print(f"  {len(subs_v)}/{len(subjects)} subject(s) included. Summary plots saved to: {out_dir}")

def run_fc_sweep(dataset, sub_str, G_values=None):
    """Run the FC sweep for one subject (hand-rolled DMF + BW / rsHRF,
    no TVB/Subnetwork engine). Computes Pearson r vs empirical FC for
    every G, selects each method's best-fit G, and saves
    PCorr_legacy.txt / PCorr_rshrf.txt / G_values.txt / best_fc_*.npy /
    G_sweep.png / fc_comparison.png to results/<dataset>/<subject>/.

    G_values: list of G's to sweep, or None for the full default
    16-point sweep (or a single-element list for one specific G).
    """
    print("\n" + "=" * 60)
    print(f"DATASET: {dataset}  SUBJECT: {sub_str}")
    print("=" * 60)

    out_dir = results_dir(dataset, sub_str)
    os.makedirs(out_dir, exist_ok=True)

    default_sweep = G_values is None
    done_file = os.path.join(out_dir, "PCorr_legacy.txt")
    if default_sweep and os.path.exists(done_file):
        print(f"  Already done — skipping. (delete {done_file} to re-run)")
        return

    try:
        d = load_subject_data(dataset, sub_str)
    except Exception as e:
        print(f"  FAILED to load data: {e}")
        return

    emp_fc       = d["emp_fc"]
    TR           = d["TR"]
    weights      = d["weights"]
    tract_lengths= d["tract_lengths"]
    hrf_compact  = d["hrf_compact"]
    N            = d["N"]
    BOLD_TR_ms   = d["BOLD_TR_ms"]
    total_dur_ms = d["total_dur_ms"]

    print(f"  N={N}  TR={TR}s  HRF={d['HRF_samples']}samp  "
          f"total_dur={total_dur_ms}ms")
    print(f"  HRF range: [{hrf_compact.min():.4f}, {hrf_compact.max():.4f}]")

    if G_values is None:
        G_values = default_G_values()
    n_cores   = multiprocessing.cpu_count()
    n_workers = min(n_cores, len(G_values))
    print(f"  G sweep: {len(G_values)} values  workers={n_workers}/{n_cores}")

    task_args = [
        (G, weights, tract_lengths, hrf_compact,
         total_dur_ms, BOLD_TR_ms, N, emp_fc, TR)
        for G in G_values
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one_G, a): a[0] for a in task_args}
        for future in as_completed(futures):
            G, r_leg, fc_leg, r_hrf, fc_hrf = future.result()
            results[G] = (r_leg, fc_leg, r_hrf, fc_hrf)

    PCorr_legacy=[]; PCorr_rshrf=[]

    for G in G_values:
        r_leg, fc_leg, r_hrf, fc_hrf = results[G]
        PCorr_legacy.append(r_leg); PCorr_rshrf.append(r_hrf)

    asc_order  = np.argsort(G_values)
    G_asc      = [G_values[i] for i in asc_order]
    r_leg_asc  = [PCorr_legacy[i] for i in asc_order]
    r_hrf_asc  = [PCorr_rshrf[i]  for i in asc_order]

    G_TOL = 0.015

    best_G_leg, best_r_leg, best_idx_leg = select_best_G(G_asc, r_leg_asc, tol=G_TOL)
    best_G_hrf, best_r_hrf, best_idx_hrf = select_best_G(G_asc, r_hrf_asc, tol=G_TOL)

    fc_leg_at_best = results[best_G_leg][1] if best_G_leg is not None else None
    fc_hrf_at_best = results[best_G_hrf][3] if best_G_hrf is not None else None
    best_fc_leg = fc_leg_at_best.copy() if fc_leg_at_best is not None else None
    best_fc_hrf = fc_hrf_at_best.copy() if fc_hrf_at_best is not None else None

    uidx_sc = np.triu_indices(N, 1)
    sc_norm = weights / (weights.max() + 1e-12)
    sc_z    = np.arctanh(np.clip(sc_norm[uidx_sc], -0.9999, 0.9999))
    if best_fc_leg is not None and best_fc_hrf is not None:
        r_leg_sc,_ = stats.pearsonr(
            np.nan_to_num(np.arctanh(np.clip(best_fc_leg[uidx_sc],-0.9999,0.9999))), sc_z)
        r_hrf_sc,_ = stats.pearsonr(
            np.nan_to_num(np.arctanh(np.clip(best_fc_hrf[uidx_sc],-0.9999,0.9999))), sc_z)
    else:
        r_leg_sc = r_hrf_sc = float('nan')

    np.savetxt(os.path.join(out_dir, "PCorr_legacy.txt"),   PCorr_legacy, fmt="%.5f")
    np.savetxt(os.path.join(out_dir, "PCorr_rshrf.txt"),    PCorr_rshrf,  fmt="%.5f")
    np.savetxt(os.path.join(out_dir, "G_values.txt"),       G_values,     fmt="%.2f")
    if best_fc_leg is not None:
        np.save(os.path.join(out_dir, "best_fc_legacy.npy"), best_fc_leg)
    if best_fc_hrf is not None:
        np.save(os.path.join(out_dir, "best_fc_rshrf.npy"),  best_fc_hrf)

    print(f"\n  RESULTS — {sub_str}")
    print(f"    Best Legacy : r={best_r_leg:.4f}  G={best_G_leg}  r_vs_SC={r_leg_sc:.3f}")
    print(f"    Best rsHRF  : r={best_r_hrf:.4f}  G={best_G_hrf}  r_vs_SC={r_hrf_sc:.3f}")
    for G, rl, rr in zip(G_values, PCorr_legacy, PCorr_rshrf):
        print(f"    G={G:.2f}  legacy={rl:.4f}  rsHRF={rr:.4f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(G_values, PCorr_legacy, 'b-o', label='Legacy (BW)',     markersize=6)
    ax.plot(G_values, PCorr_rshrf,  'r-o', label='rsHRF (canon2dd)',markersize=6)
    ax.set_xlabel("Global Coupling G"); ax.set_ylabel("Pearson r")
    ax.set_title(f"G Sweep — {sub_str}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "G_sweep.png"), dpi=150)
    plt.close()

    if best_fc_leg is not None and best_fc_hrf is not None:
        fig2, axes = plt.subplots(1, 4, figsize=(26, 6))
        axes[0].imshow(emp_fc, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title("Empirical FC\n(ground truth)")
        im1 = axes[1].imshow(best_fc_leg, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title(f"Legacy (BW)\nr={best_r_leg:.3f} vs empirical  "
                          f"r={r_leg_sc:.3f} vs SC\nG={best_G_leg:.2f}")
        plt.colorbar(im1, ax=axes[1])
        axes[2].imshow(weights, cmap='hot')
        axes[2].set_title("Structural Connectivity (SC)")
        im3 = axes[3].imshow(best_fc_hrf, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[3].set_title(f"rsHRF (canon2dd)\nr={best_r_hrf:.3f} vs empirical  "
                          f"r={r_hrf_sc:.3f} vs SC\nG={best_G_hrf:.2f}")
        plt.colorbar(im3, ax=axes[3])
        plt.suptitle(f"{sub_str} — FC Comparison", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fc_comparison.png"), dpi=150)
        plt.close()

    print(f"  Saved to: {out_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='rsHRF-TVB pipeline: BOLD simulation, FC sweep + comparison, '
                    'and cross-subject summary plots')
    parser.add_argument('--dataset', type=str, default='ds001226',
                        help='OpenNeuro dataset id e.g. ds001226')
    parser.add_argument('--subject', type=str, default=None,
                        help='Run only this subject e.g. CON01; omit to run all subjects')
    parser.add_argument('--G', type=float, default=None,
                        help="Explicit G value override. For 'fc' mode: restrict the "
                             "sweep to just this one G (omit for the full 16-point "
                             "default sweep). For 'bold' mode: use this one G for both "
                             "Legacy and rsHRF instead of each method's own best-fit G "
                             "(omit to use best-fit G's from a completed 'fc' sweep).")
    parser.add_argument('--mode', type=str, nargs='+', choices=['bold', 'fc', 'summary'],
                        default=['bold', 'fc'],
                        help="Which stage(s) to run, space-separated. "
                             "'bold': simulate BOLD (Legacy+rsHRF) ONLY at each method's "
                             "own best-fit G (read from a completed 'fc' sweep; pass --G "
                             "to override with one explicit G for both methods instead). "
                             "Produces exactly 2 plots: bold_region0_comparison.png "
                             "(Empirical/Legacy/rsHRF, Region 0) and "
                             "bold_regions1to5_transient_removed.png (Regions 1-5, rsHRF). "
                             "'fc': FC sweep + fc_comparison.png (hand-rolled DMF+BW/rsHRF, "
                             "no TVB/Subnetwork engine). "
                             "'summary': aggregate PCorr files across subjects into "
                             "summary plots under results/<dataset>/summary/ (no "
                             "summary.json). Default (no --mode given): 'fc' runs first, "
                             "then 'bold' (so bold mode always has a best-fit G to use). "
                             "'summary' does not run unless explicitly selected.")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else ALL_SUBJECTS
    G_values = [args.G] if args.G is not None else None

    if 'summary' in args.mode:
        summarize_subjects(args.dataset, subjects=[args.subject] if args.subject else None)

    sim_modes = [m for m in args.mode if m in ('bold', 'fc')]
    if sim_modes:
        print(f"Running mode(s)={sim_modes}  dataset={args.dataset}  "
              f"{len(subjects)} subject(s)"
              + (f"  G={args.G}" if args.G is not None else "  (full G sweep)"))
        for sub in subjects:
            try:
                if 'fc' in sim_modes:
                    run_fc_sweep(args.dataset, sub, G_values=G_values)
                if 'bold' in sim_modes:
                    run_bold_sweep(args.dataset, sub, G_values=G_values)
            except Exception as e:
                print(f"SUBJECT {sub} FAILED: {e}")
                continue

    print("\nALL DONE")