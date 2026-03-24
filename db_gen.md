# Génération du dataset OFI

Script : `generate_dataset.py`
Génère des signaux OFI (Optical Feedback Interferometry) simulés pour entraîner un CNN de classification de particules (2 µm, 4 µm, 10 µm).

## Prérequis

```bash
pip install numpy scipy tqdm
```

Le dossier `Noise/` (305 fichiers `.npy` de bruit réel) est nécessaire pour `--noise real`.

---

## Exemples rapides

```bash
# Signal pur, pas de bruit (défaut)
python generate_dataset.py test --output ./preview --force

# Signal réaliste + bruit coloré
python generate_dataset.py auto --signal realistic --noise colored --force

# Tout réaliste (signal + bruit avant filtre + amplitude variable)
python generate_dataset.py auto --signal realistic --noise realistic --force

# Injection de vrai bruit
python generate_dataset.py auto --noise real --force

# Override d'un paramètre via config
echo -e "[noise]\nnoise_sigma = 0.1" > custom.ini
python generate_dataset.py auto --config custom.ini --force
```

---

## Les 3 modes

| Mode | Description | Commande |
|------|------------|---------|
| `auto` | Dataset complet (1511 fichiers, train/test split) | `... auto` |
| `test` | Aperçu rapide (3 signaux/classe, dossier plat) | `... test` |
| `manual` | Config `.ini` obligatoire | `... manual --config params.ini` |

---

## Options communes

| Option | Défaut | Description |
|---|---|---|
| `--signal {pure,realistic}` | `realistic` | Preset signal |
| `--noise {none,white,colored,realistic,real}` | `none` | Preset bruit |
| `--noise-dir DIR` | `./Noise` | Dossier des fichiers de bruit réel |
| `--config FILE` | — | Fichier `.ini` pour surcharger les paramètres |
| `--init-config FILE` | — | Génère un template `.ini` et quitte |
| `--output DIR` | `./dataset` ou `./dataset_test` | Dossier de sortie |
| `--force` | — | Écrase le dossier existant |

### Résolution des paramètres

```
DEFAULT_SIM  →  preset --signal  →  preset --noise  →  fichier --config (gagne)
```

---

## Presets signal (`--signal`)

| Preset | dc_offset_std | multiburst_pct | envelope_skew |
|--------|--------------|----------------|---------------|
| `pure` | 0 | 0 | [0, 0] |
| `realistic` (défaut) | 0.15 | 10% | [-0.5, 0.5] |

### Détail des paramètres signal (`[signal]` dans le `.ini`)

| Paramètre | Description |
|---|---|
| `dc_offset_std` | Écart-type de l'offset DC aléatoire (0 = désactivé) |
| `multiburst_pct` | % de signaux avec un second burst (0 = désactivé) |
| `envelope_skew_min` | Borne inf de l'asymétrie de l'enveloppe |
| `envelope_skew_max` | Borne sup (0/0 = gaussienne symétrique) |

---

## Presets bruit (`--noise`)

| Preset | noise_type | noise_injection | noise_sigma | noise_variability |
|--------|-----------|-----------------|-------------|-------------------|
| `none` (défaut) | — | — | 0 | 0 |
| `white` | blanc gaussien | après filtre | 0.058 | 0 |
| `colored` | coloré dual-band (1–80 kHz) | après filtre | 0.058 | 0 |
| `realistic` | blanc | **avant filtre** | 0.21 | 19% |
| `real` | fichiers réels | après filtre | 0.058 | 0 |

### Détail des paramètres bruit (`[noise]` dans le `.ini`)

| Paramètre | Valeurs | Description |
|---|---|---|
| `noise_type` | `none`, `white`, `colored`, `real` | Type de bruit |
| `noise_injection` | `before`, `after` | Point d'injection vs bandpass |
| `noise_sigma` | float | Écart-type du bruit |
| `noise_variability` | float (CV) | Variabilité d'amplitude inter-samples (0 = fixe) |

### `noise_injection = before`

Le bruit est ajouté AVANT le filtre passe-bande → il est filtré naturellement comme dans le réel. Le sigma doit être plus élevé (~0.21) car le filtre élimine la puissance hors bande.

### `noise_variability`

Pour chaque sample, sigma est modulé par `lognormal(0, noise_variability)`. La valeur 0.19 reproduit les 19% de coefficient de variation mesurés sur les vrais fichiers de bruit.

---

## Pipeline de génération

```
1. simulated_particle()       → signal brut
2. [multiburst]               → ajout éventuel d'un 2e burst
3. [envelope skew]            → asymétrie de l'enveloppe
4. DC subtraction             → suppression du DC
5. [noise if before]          → bruit AVANT bandpass
6. bandpass filter             → filtre passe-bande 7–80 kHz
7. [noise if after]           → bruit APRÈS bandpass
8. [DC offset]                → offset aléatoire
```

- Étapes 2, 3, 8 contrôlées par `--signal`
- Étapes 5, 7 contrôlées par `--noise`
- Signal et bruit sont **indépendants**

---

## Paramètres par défaut

### Simulation physique

| Paramètre | Valeur | Description |
|---|---|---|
| `laser_lambda` | 1550e-9 m | Longueur d'onde |
| `adq_freq` | 2 MHz | Fréquence d'acquisition |
| `inc_angle` | 80° | Angle d'incidence |
| `po` | 0.016536 mV | Puissance laser |
| `time_max` | 2500 | Échantillons par signal |
| `s_l` | 7e-6 m | Diamètre du spot laser |
| `p_speed` | [0.05, 0.20] m/s | Vitesse de la particule |
| `t_impact` | [40%, 60%] de la fenêtre | Position du burst |

### Filtre

| Paramètre | Valeur |
|---|---|
| `filter_lowcut` | 7 000 Hz |
| `filter_highcut` | 80 000 Hz |
| `filter_order` | 4 (Butterworth) |

### Indice de modulation par classe (m0)

| Classe | m0 min | m0 max | Ratio d'amplitude |
|---|---|---|---|
| 2 µm | 7.0 | 14.0 | 1.00x |
| 4 µm | 18.0 | 36.0 | ~1.86x |
| 10 µm | 20.0 | 95.0 | ~4.33x |

---

## Fichier de config `.ini`

Sections disponibles :

| Section | Contenu |
|---|---|
| `[simulation]` | Paramètres physiques |
| `[randomization]` | Plages de vitesse, T_impact |
| `[postprocessing]` | Filtre passe-bande |
| `[noise]` | Type, injection, sigma, variabilité |
| `[signal]` | DC offset, multiburst, envelope skew |
| `[class_NOM]` | Taille particule, train/test, m0 min/max |

Générer un template complet :
```bash
python generate_dataset.py auto --init-config params.ini
```

---

## Interface graphique (Streamlit)

Alternative visuelle au CLI. Tous les paramètres sont visibles et modifiables avant de lancer la génération.

```bash
streamlit run generate_ui.py
```

Fonctionnalités :
- Dropdowns pour les presets signal et bruit (auto-remplissage des champs)
- Tous les paramètres éditables en temps réel
- Preview : génère 3 samples/classe et affiche les plots + statistiques
- Commande CLI équivalente affichée en bas (pour apprendre le CLI)
- Sections avancées dépliables (simulation physique, filtre, classes)
- Bouton de génération complète avec barre de progression

---

## Format des fichiers générés

- `.npy` (NumPy binary), shape `(2500,)`, dtype `float64`
- Nommage : `sample_0000.npy`, `sample_0001.npy`, …
