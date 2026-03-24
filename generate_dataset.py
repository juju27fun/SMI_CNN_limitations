"""
generate_dataset.py
-------------------
Génère un dataset CNN pour la classification de particules OFI (2 µm, 4 µm, 10 µm).

Modes:
    auto    Paramètres par défaut, dataset complet (1511 fichiers)
    test    3 signaux par classe dans un dossier plat (pour inspection visuelle)
    manual  Paramètres personnalisés depuis un fichier .ini

Usage:
    python generate_dataset.py auto [--output ./dataset] [--force]
    python generate_dataset.py test [--output ./dataset_test] [--force]
    python generate_dataset.py manual --config params.ini [--output ./dataset] [--force]
    python generate_dataset.py manual --init-config params.ini
"""

import argparse
import configparser
import contextlib
import io
import os
import shutil

import glob as glob_mod

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.special import erf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Copie inline de simulated_particle() — NE PAS importer Simulated_Particle.py
# (le code module-level appelle plt.show() qui bloquerait l'exécution)
# ---------------------------------------------------------------------------
def simulated_particle(P_size, P_Speed, Inc_Angle, Laser_Lambda, Po, T_impact,
                       S_l, Time_max, Adq_Freq, m0):
    """
    Simulate the signal of a single particle based on Optical Feedback Interferometry.

    Parameters
    -----------
    P_size      : Diameter of the particle [m]
    P_Speed     : Particle speed [m/s]
    Inc_Angle   : Incident angle [degrees]
    Laser_Lambda: Laser wavelength [m]
    Po          : Laser power [mV]
    T_impact    : Time when particle center crosses beam center [s]
    S_l         : Laser beam spot diameter [m]
    Time_max    : Number of samples
    Adq_Freq    : Acquisition frequency [Hz]
    m0          : Modulation index

    Returns
    -----------
    P_t, t      : Signal vector [mV], Time vector [s]
    """
    Inc_Angle_rad = np.radians(Inc_Angle)
    Inc_Angle_rad_2 = np.radians(90 - Inc_Angle)

    t = np.linspace(0, Time_max / Adq_Freq, Time_max)

    f_D = (2 * P_Speed * np.sin(Inc_Angle_rad_2)) / Laser_Lambda
    print('doppler:', f_D)
    print('P_Speed:', P_Speed)

    tau = (P_size + S_l) / (P_Speed * np.sin(Inc_Angle_rad))
    print('tau:', tau)

    modulation = 1 + m0 * np.cos(2 * np.pi * f_D * t)
    envelope = np.exp(-((t - T_impact) ** 2) / (2 * tau ** 2))
    P_t = Po * modulation * envelope

    return P_t, t


# ---------------------------------------------------------------------------
# Filtre passe-bande (reproduit le traitement réel du Bokeh viewer)
# ---------------------------------------------------------------------------
def bandpass_filter(data, lowcut, highcut, order, fs):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, data)


# ---------------------------------------------------------------------------
# Chargement des fichiers de bruit réel
# ---------------------------------------------------------------------------
def load_noise_files(noise_dir):
    """Pre-load all .npy noise files from directory. Returns list of np arrays."""
    paths = sorted(glob_mod.glob(os.path.join(noise_dir, "*.npy")))
    if not paths:
        raise FileNotFoundError(
            f"Aucun fichier .npy trouvé dans '{noise_dir}'. "
            "Vérifiez le chemin ou utilisez --noise generated."
        )
    noise_arrays = [np.load(p) for p in paths]
    print(f"Bruit réel : {len(noise_arrays)} fichiers chargés depuis '{noise_dir}'")
    return noise_arrays


# ---------------------------------------------------------------------------
# Génération de bruit coloré synthétique
# ---------------------------------------------------------------------------
def generate_colored_noise(n_samples, rng, sigma, fs):
    """Generate colored noise with shaped PSD (70% power 1–10 kHz, 30% 10–80 kHz)."""
    white1 = rng.normal(0, 1.0, n_samples)
    white2 = rng.normal(0, 1.0, n_samples)
    low  = bandpass_filter(white1, lowcut=1000,  highcut=10000, order=4, fs=fs)
    high = bandpass_filter(white2, lowcut=10000, highcut=80000, order=4, fs=fs)
    colored = np.sqrt(0.7) * low + np.sqrt(0.3) * high
    std = np.std(colored)
    if std > 0:
        colored = colored * (sigma / std)
    return colored


# ---------------------------------------------------------------------------
# Fabrication du bruit (dispatch type + variabilité)
# ---------------------------------------------------------------------------
def _get_real_noise(noise_files, rng, n_samples, sigma):
    """Extract a random segment from real noise files and scale to sigma."""
    noise_sig = noise_files[rng.integers(len(noise_files))]
    offset = rng.integers(0, len(noise_sig) - n_samples)
    noise = noise_sig[offset:offset + n_samples].copy()
    noise = noise - np.mean(noise)
    std = np.std(noise)
    if std > 0:
        noise = noise * (sigma / std)
    return noise


def _make_noise(sim, rng, noise_files):
    """Generate a noise vector according to sim params. Returns None if no noise."""
    if sim["noise_type"] == "none" or sim["noise_sigma"] <= 0:
        return None
    sigma = sim["noise_sigma"]
    if sim["noise_variability"] > 0:
        sigma = sigma * rng.lognormal(0, sim["noise_variability"])
    if sim["noise_type"] == "white":
        return rng.normal(0, sigma, sim["time_max"])
    elif sim["noise_type"] == "colored":
        return generate_colored_noise(sim["time_max"], rng, sigma, sim["adq_freq"])
    elif sim["noise_type"] == "real":
        return _get_real_noise(noise_files, rng, sim["time_max"], sigma)
    return None


# ---------------------------------------------------------------------------
# Paramètres par défaut
# ---------------------------------------------------------------------------
DEFAULT_SIM = {
    # Simulation physique
    "laser_lambda":           1550e-9,
    "adq_freq":               2_000_000,
    "inc_angle":              80.0,
    "po":                     0.004134 * 4,
    "time_max":               2500,
    "s_l":                    7e-6,
    # Randomisation par sample
    "p_speed_min":            0.05,
    "p_speed_max":            0.20,
    "t_impact_factor_min":    0.4,
    "t_impact_factor_max":    0.6,
    # Post-traitements
    "filter_lowcut":          7000,
    "filter_highcut":         80000,
    "filter_order":           4,
    # Bruit (contrôlés par --noise, surchargeables via .ini [noise])
    "noise_type":             "none",
    "noise_injection":        "after",
    "noise_sigma":            0.0,
    "noise_variability":      0.0,
    # Signal (contrôlés par --signal, surchargeables via .ini [signal])
    "dc_offset_std":          0.15,
    "multiburst_pct":         10,
    "envelope_skew_min":      -0.5,
    "envelope_skew_max":      0.5,
    # Reproductibilité
    "seed":                   42,
}

NOISE_PRESETS = {
    "none":      {"noise_type": "none",    "noise_injection": "after",  "noise_sigma": 0.0,   "noise_variability": 0.0},
    "white":     {"noise_type": "white",   "noise_injection": "after",  "noise_sigma": 0.058, "noise_variability": 0.0},
    "colored":   {"noise_type": "colored", "noise_injection": "after",  "noise_sigma": 0.058, "noise_variability": 0.0},
    "realistic": {"noise_type": "white",   "noise_injection": "before", "noise_sigma": 0.21,  "noise_variability": 0.19},
    "real":      {"noise_type": "real",    "noise_injection": "after",  "noise_sigma": 0.058, "noise_variability": 0.0},
}

SIGNAL_PRESETS = {
    "pure": {
        "dc_offset_std":      0.0,
        "multiburst_pct":     0,
        "envelope_skew_min":  0.0,
        "envelope_skew_max":  0.0,
    },
    "realistic": {
        "dc_offset_std":      0.15,
        "multiburst_pct":     10,
        "envelope_skew_min":  -0.5,
        "envelope_skew_max":  0.5,
    },
}

DEFAULT_CLASSES = [
    {"name": "2um",  "p_size": 2e-6,  "train": 403, "test": 101, "m0_min": 7.0,  "m0_max": 14.0},
    {"name": "4um",  "p_size": 4e-6,  "train": 403, "test": 101, "m0_min": 18.0, "m0_max": 36.0},
    {"name": "10um", "p_size": 10e-6, "train": 403, "test": 100, "m0_min": 20.0, "m0_max": 95.0},
]

TEMPLATE_CONFIG = """\
# Configuration pour generate_dataset.py (mode manual)
# Tous les paramètres sont optionnels : les valeurs absentes prennent la valeur par défaut.

[simulation]
# Longueur d'onde du laser [m]
laser_lambda = 1550e-9
# Fréquence d'acquisition [Hz]
adq_freq = 2000000
# Angle d'incidence [degrés]
inc_angle = 80.0
# Puissance laser [mV]  (Po = 0.004134 * 4)
po = 0.016536
# Nombre d'échantillons par signal
time_max = 2500
# Diamètre du spot laser [m]
s_l = 7e-6
# Graine pour la reproductibilité
seed = 42

[randomization]
# Vitesse de la particule [m/s]  — tirée uniformément dans [min, max]
# Contrainte : f_D = 2*V*sin(10°)/lambda doit rester > 7000 Hz (coupure basse du filtre)
# À 0.05 m/s → f_D ≈ 11.2 kHz  (minimum recommandé)
p_speed_min = 0.05
p_speed_max = 0.20
# Position du centre de la particule dans la fenêtre temporelle (fraction de window)
t_impact_factor_min = 0.4
t_impact_factor_max = 0.6

[postprocessing]
# Filtre passe-bande Butterworth (ordre, fréquences de coupure en Hz)
filter_lowcut  = 7000
filter_highcut = 80000
filter_order   = 4

[noise]
# Type de bruit : none, white, colored, real
noise_type = colored
# Point d'injection : before (avant filtre) ou after (après filtre)
noise_injection = after
# Écart-type du bruit (calibré sur données réelles)
noise_sigma = 0.058
# Variabilité d'amplitude inter-samples (CV, 0 = fixe, 0.19 = réaliste)
noise_variability = 0.0

[signal]
# Paramètres du signal (contrôlés par --signal preset, surchargeables ici)
# Offset DC aléatoire (σ en mV, 0 pour désactiver)
dc_offset_std = 0.15
# Pourcentage de signaux avec un second burst (0 pour désactiver)
multiburst_pct = 10
# Plage d'asymétrie de l'enveloppe gaussienne (0 0 pour symétrique)
envelope_skew_min = -0.5
envelope_skew_max = 0.5

# Une section [class_NOM] par classe à générer.
# p_size  : diamètre de la particule [m]
# train   : nombre de samples d'entraînement
# test    : nombre de samples de test
# m0_min  : indice de modulation minimum (plus grand pour les grosses particules)
# m0_max  : indice de modulation maximum

[class_2um]
p_size = 2e-6
train  = 403
test   = 101
m0_min = 7.0
m0_max = 14.0

[class_4um]
p_size = 4e-6
train  = 403
test   = 101
m0_min = 18.0
m0_max = 36.0

[class_10um]
p_size = 10e-6
train  = 403
test   = 100
m0_min = 20.0
m0_max = 95.0
"""


# ---------------------------------------------------------------------------
# Lecture du fichier de configuration .ini
# ---------------------------------------------------------------------------
def load_config_file(path, base_sim=None):
    cfg = configparser.ConfigParser()
    cfg.read(path)

    sim = dict(base_sim if base_sim is not None else DEFAULT_SIM)

    if "simulation" in cfg:
        s = cfg["simulation"]
        for key in ("laser_lambda", "adq_freq", "inc_angle", "po", "s_l"):
            if key in s:
                sim[key] = float(s[key])
        if "time_max" in s:
            sim["time_max"] = int(float(s["time_max"]))
        if "seed" in s:
            sim["seed"] = int(float(s["seed"]))

    if "randomization" in cfg:
        r = cfg["randomization"]
        for key in ("p_speed_min", "p_speed_max", "m0_min", "m0_max",
                    "t_impact_factor_min", "t_impact_factor_max"):
            if key in r:
                sim[key] = float(r[key])

    if "postprocessing" in cfg:
        p = cfg["postprocessing"]
        for key in ("filter_lowcut", "filter_highcut"):
            if key in p:
                sim[key] = float(p[key])
        if "filter_order" in p:
            sim["filter_order"] = int(float(p["filter_order"]))
        # Rétro-compat : noise_sigma et dc_offset_std dans [postprocessing]
        if "noise_sigma" in p:
            sim["noise_sigma"] = float(p["noise_sigma"])
        if "dc_offset_std" in p:
            sim["dc_offset_std"] = float(p["dc_offset_std"])

    if "noise" in cfg:
        n = cfg["noise"]
        for key in ("noise_sigma", "noise_variability"):
            if key in n:
                sim[key] = float(n[key])
        for key in ("noise_type", "noise_injection"):
            if key in n:
                sim[key] = n[key]

    if "signal" in cfg:
        s = cfg["signal"]
        for key in ("dc_offset_std", "envelope_skew_min", "envelope_skew_max"):
            if key in s:
                sim[key] = float(s[key])
        if "multiburst_pct" in s:
            sim["multiburst_pct"] = float(s["multiburst_pct"])

    classes = []
    for section in cfg.sections():
        if section.startswith("class_"):
            name = section[len("class_"):]
            c = cfg[section]
            cls_entry = {
                "name":   name,
                "p_size": float(c["p_size"]),
                "train":  int(c.get("train", "403")),
                "test":   int(c.get("test",  "101")),
            }
            if "m0_min" in c:
                cls_entry["m0_min"] = float(c["m0_min"])
            if "m0_max" in c:
                cls_entry["m0_max"] = float(c["m0_max"])
            classes.append(cls_entry)

    if not classes:
        classes = list(DEFAULT_CLASSES)

    return sim, classes


# ---------------------------------------------------------------------------
# Génération d'un sample (paramétrisée)
# ---------------------------------------------------------------------------
def generate_sample(p_size, rng, sim, m0_min, m0_max, noise_files=None):
    """
    Retourne (signal, params) :
      signal : np.ndarray float64 shape (time_max,)
      params : dict avec les valeurs tirées aléatoirement (pour le log en mode test)
    """
    window   = sim["time_max"] / sim["adq_freq"]
    p_speed  = rng.uniform(sim["p_speed_min"],         sim["p_speed_max"])
    m0       = rng.uniform(m0_min,                      m0_max)
    t_impact = rng.uniform(sim["t_impact_factor_min"] * window,
                           sim["t_impact_factor_max"] * window)

    _sim_args = dict(
        P_size=p_size, P_Speed=p_speed, Inc_Angle=sim["inc_angle"],
        Laser_Lambda=sim["laser_lambda"], Po=sim["po"], S_l=sim["s_l"],
        Time_max=sim["time_max"], Adq_Freq=sim["adq_freq"],
    )

    with contextlib.redirect_stdout(io.StringIO()):
        P_t, _ = simulated_particle(T_impact=t_impact, m0=m0, **_sim_args)

    # --- Multiburst (avant DC subtraction) ---
    is_multiburst = False
    if sim["multiburst_pct"] > 0 and rng.random() < sim["multiburst_pct"] / 100:
        is_multiburst = True
        p_speed2 = rng.uniform(sim["p_speed_min"], sim["p_speed_max"])
        m0_2 = rng.uniform(m0_min, m0_max)
        margin = 0.15 * window
        lo = sim["t_impact_factor_min"] * window
        hi = sim["t_impact_factor_max"] * window
        if t_impact < window / 2:
            t2_lo, t2_hi = min(t_impact + margin, hi), hi
        else:
            t2_lo, t2_hi = lo, max(t_impact - margin, lo)
        if t2_lo < t2_hi:
            t_impact2 = rng.uniform(t2_lo, t2_hi)
            with contextlib.redirect_stdout(io.StringIO()):
                P_t2, _ = simulated_particle(
                    T_impact=t_impact2, m0=m0_2,
                    **{**_sim_args, "P_Speed": p_speed2},
                )
            P_t = P_t + P_t2

    # --- Envelope skew (avant DC subtraction) ---
    skew = rng.uniform(sim["envelope_skew_min"], sim["envelope_skew_max"])
    if skew != 0:
        t = np.linspace(0, window, sim["time_max"])
        tau = (p_size + sim["s_l"]) / (p_speed * np.sin(np.radians(sim["inc_angle"])))
        z = (t - t_impact) / tau
        skew_factor = 1 + erf(skew * z / np.sqrt(2))
        P_t = P_t * skew_factor

    # --- DC subtraction ---
    P_t = P_t - np.mean(P_t)

    # --- Bruit AVANT bandpass (injection=before) ---
    noise = _make_noise(sim, rng, noise_files)
    if noise is not None and sim["noise_injection"] == "before":
        P_t = P_t + noise

    # --- Bandpass ---
    P_t = bandpass_filter(P_t, sim["filter_lowcut"], sim["filter_highcut"],
                          sim["filter_order"], sim["adq_freq"])

    # --- Bruit APRÈS bandpass (injection=after) ---
    if noise is not None and sim["noise_injection"] == "after":
        P_t = P_t + noise

    # --- DC offset (contrôlé par --signal, indépendant du bruit) ---
    dc_offset = 0.0
    if sim["dc_offset_std"] > 0:
        dc_offset = rng.normal(0, sim["dc_offset_std"])
        P_t = P_t + dc_offset

    params = {
        "p_speed": p_speed, "m0": m0, "t_impact": t_impact,
        "dc_offset": dc_offset, "skew": skew, "multiburst": is_multiburst,
    }
    return P_t.astype(np.float64), params


# ---------------------------------------------------------------------------
# Ajout automatique du dataset au .gitignore
# ---------------------------------------------------------------------------
def _add_to_gitignore(output_dir):
    """Add output_dir to .gitignore (create the file if it doesn't exist)."""
    entry = os.path.normpath(output_dir).strip(os.sep)
    if not entry.endswith("/"):
        entry += "/"
    gitignore_path = os.path.join(os.path.dirname(output_dir) or ".", ".gitignore")
    if os.path.isfile(gitignore_path):
        existing = open(gitignore_path).read()
        if entry in existing.splitlines():
            return
        with open(gitignore_path, "a") as f:
            if not existing.endswith("\n"):
                f.write("\n")
            f.write(entry + "\n")
    else:
        with open(gitignore_path, "w") as f:
            f.write(entry + "\n")
    print(f".gitignore : ajouté '{entry}'")


# ---------------------------------------------------------------------------
# Moteur de génération commun aux 3 modes
# ---------------------------------------------------------------------------
def run_generation(classes, sim, output_dir, force, test_mode=False,
                   noise_dir="./Noise"):
    if os.path.exists(output_dir):
        if force:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Le dossier '{output_dir}' existe déjà. "
                "Utilisez --force pour l'écraser."
            )

    np.random.seed(sim["seed"])
    rng   = np.random.default_rng(sim["seed"])
    total = 0

    # Chargement des fichiers de bruit réel si nécessaire
    noise_files = None
    if sim["noise_type"] == "real":
        noise_files = load_noise_files(noise_dir)

    # Fallback: si m0 n'est pas défini par classe, utiliser la valeur globale
    for cls in classes:
        if "m0_min" not in cls:
            cls["m0_min"] = sim.get("m0_min", 8.0)
        if "m0_max" not in cls:
            cls["m0_max"] = sim.get("m0_max", 14.0)

    if test_mode:
        # --- mode test : 3 signaux par classe, dossier plat ---
        for cls in classes:
            folder = os.path.join(output_dir, cls["name"])
            os.makedirs(folder, exist_ok=True)
            open(os.path.join(folder, ".gitkeep"), "w").close()

            print(f"\n--- {cls['name']}  (P_size = {cls['p_size']*1e6:.0f} µm) ---")
            for i in range(3):
                signal, params = generate_sample(
                    cls["p_size"], rng, sim, cls["m0_min"], cls["m0_max"],
                    noise_files=noise_files,
                )
                np.save(os.path.join(folder, f"sample_{i:04d}.npy"), signal)
                tau = (cls["p_size"] + sim["s_l"]) / (
                    params["p_speed"] * np.sin(np.radians(sim["inc_angle"]))
                )
                print(
                    f"  sample_{i:04d}.npy  |  "
                    f"P_Speed={params['p_speed']:.4f} m/s  |  "
                    f"m0={params['m0']:.2f}  |  "
                    f"T_impact={params['t_impact']*1000:.3f} ms  |  "
                    f"tau={tau*1e6:.1f} µs"
                )
                total += 1

        print(f"\nTotal : {total} fichiers dans '{output_dir}'")

    else:
        # --- modes auto / manual : dataset complet avec split train/test ---
        for cls in classes:
            for split, n_samples in [("train", cls["train"]), ("test", cls["test"])]:
                folder = os.path.join(output_dir, split, cls["name"])
                os.makedirs(folder, exist_ok=True)
                open(os.path.join(folder, ".gitkeep"), "w").close()

                for i in tqdm(range(n_samples), desc=f"{split}/{cls['name']}", unit="sample"):
                    signal, _ = generate_sample(
                        cls["p_size"], rng, sim, cls["m0_min"], cls["m0_max"],
                        noise_files=noise_files,
                    )
                    np.save(os.path.join(folder, f"sample_{i:04d}.npy"), signal)
                total += n_samples

        for cls in classes:
            print(f"[{cls['name']:4s}]  train={cls['train']}  test={cls['test']}")
        print(f"Total : {total} fichiers générés dans '{output_dir}'")

    _add_to_gitignore(output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Génère un dataset CNN de signaux OFI simulés.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
            "  python generate_dataset.py auto\n"
            "  python generate_dataset.py test --output ./preview\n"
            "  python generate_dataset.py manual --init-config params.ini\n"
            "  python generate_dataset.py manual --config params.ini\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_args(sub):
        sub.add_argument("--output", default="./dataset", help="Dossier de sortie")
        sub.add_argument("--force", action="store_true",
                         help="Écrase le dossier de sortie s'il existe déjà")
        sub.add_argument("--signal", choices=["pure", "realistic"], default="realistic",
                         help="Preset signal : 'pure' ou 'realistic' (défaut)")
        sub.add_argument("--noise", choices=["none", "white", "colored", "realistic", "real"],
                         default="none",
                         help="Preset bruit : none (défaut), white, colored, realistic, real")
        sub.add_argument("--noise-dir", default="./Noise", metavar="DIR",
                         help="Dossier des fichiers de bruit réel (pour --noise real)")
        sub.add_argument("--config", metavar="FILE",
                         help="Fichier .ini pour surcharger les paramètres par défaut")
        sub.add_argument("--init-config", metavar="FILE",
                         help="Génère un fichier de config template et quitte")

    # --- auto ---
    p_auto = subparsers.add_parser(
        "auto", help="Dataset complet avec les paramètres par défaut (1511 fichiers)"
    )
    add_common_args(p_auto)
    p_auto.set_defaults(output="./dataset")

    # --- test ---
    p_test = subparsers.add_parser(
        "test", help="3 signaux par classe pour inspection visuelle rapide"
    )
    add_common_args(p_test)
    p_test.set_defaults(output="./dataset_test")

    # --- manual ---
    p_manual = subparsers.add_parser(
        "manual", help="Paramètres personnalisés depuis un fichier .ini"
    )
    add_common_args(p_manual)

    args = parser.parse_args()

    # Cas spécial : génération du template de config (disponible dans tous les modes)
    if args.init_config:
        with open(args.init_config, "w") as f:
            f.write(TEMPLATE_CONFIG)
        print(f"Template de configuration écrit dans '{args.init_config}'")
        return

    # En mode manual, --config est obligatoire
    if args.mode == "manual" and not args.config:
        parser.error(
            "--config est requis en mode manual "
            "(ou utilisez --init-config FILE pour créer un template)"
        )

    # Résolution : DEFAULT_SIM → preset signal → preset noise → config .ini (gagne)
    base_sim = dict(DEFAULT_SIM)
    base_sim.update(SIGNAL_PRESETS[args.signal])
    base_sim.update(NOISE_PRESETS[args.noise])

    if args.config:
        sim, classes = load_config_file(args.config, base_sim=base_sim)
    else:
        sim, classes = base_sim, list(DEFAULT_CLASSES)

    test_mode = args.mode == "test"
    run_generation(classes, sim, args.output, args.force, test_mode=test_mode,
                   noise_dir=args.noise_dir)


if __name__ == "__main__":
    main()
