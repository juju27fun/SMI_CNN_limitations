# Benchmark Datasets — Design & Justification

Ce document décrit les datasets utilisés dans le benchmark OFI, justifie l'intérêt de chacun, et détaille les analyses comparatives clés que la matrice de cross-testing permettra d'extraire.

---

## 1. Vue d'ensemble

12 datasets au total : 10 synthétiques (S0–S9), 1 union (S_union), 1 réel (dataset).

Tous les datasets synthétiques partagent les mêmes paramètres de base :
- **1511 fichiers** (1209 train / 302 test), split 80/20
- **3 classes** : 2 µm (403+101), 4 µm (403+101), 10 µm (403+100)
- **Seed = 42** pour reproductibilité
- **Même modèle physique** OFI (Doppler + enveloppe Gaussienne)

### Grille signal × bruit complète

|              | none | white | colored | realistic | real |
|--------------|------|-------|---------|-----------|------|
| **pure**     | S0   | S1    | S2 (+ S8, S9) | S6 | S7 |
| **realistic**| S5   | —     | —       | S3        | S4   |

---

## 2. Tier 1 — Core (gradient de difficulté)

### S0_baseline — `--signal pure --noise none`

**Question :** Le modèle apprend-il les features discriminantes de base ?

**Justification :** Borne supérieure théorique. Les signaux sont des cosinus Doppler purs modulés par une enveloppe Gaussienne symétrique, sans aucune perturbation. Si un modèle échoue sur S0, l'architecture est inadéquate. La performance sur S0 quantifie la difficulté intrinsèque de la tâche (overlap des classes en amplitude/fréquence).

**Hypothèse :** Accuracy > 95%. Les classes sont séparables par la fréquence Doppler et l'indice de modulation m0.

### S1_white — `--signal pure --noise white` (σ=0.058)

**Question :** Le modèle résiste-t-il au bruit blanc additif ?

**Justification :** Premier niveau de perturbation. Le bruit blanc a une PSD plate, ce qui est le modèle de bruit le plus simple et le mieux compris théoriquement. Le σ=0.058 est calibré sur les mesures réelles (écart-type dans les régions silencieuses des signaux acquis).

**Hypothèse :** Perte de 2–5% vs S0. Le bruit blanc est facile à filtrer (le bandpass training-time F2 supprime le hors-bande).

### S2_colored — `--signal pure --noise colored` (σ=0.058)

**Question :** Le bruit coloré (PSD réaliste) dégrade-t-il plus que le bruit blanc ?

**Justification :** Le bruit réel n'est pas blanc — il concentre plus d'énergie dans les basses fréquences (1/f, bruit d'amplificateur). Le preset colored utilise un split 70/30 (1–10 kHz / 10–80 kHz) qui approxime la PSD mesurée. Comparer S1 vs S2 isole l'effet de la forme spectrale du bruit à amplitude totale constante.

**Hypothèse :** S2 légèrement plus dur que S1 car le bruit coloré a plus d'énergie dans la bande du signal (7–80 kHz).

### S3_realistic — `--signal realistic --noise realistic`

**Question :** Quelle est la performance en conditions synthétiques les plus réalistes ?

**Justification :** Combine toutes les distorsions signal (DC offset σ=0.15, 10% multiburst, skew ±0.5) avec le bruit le plus fidèle (injection avant filtre, σ_base=0.21, variabilité inter-échantillon CV=0.19). C'est le dataset synthétique le plus proche des conditions réelles.

**Hypothèse :** Perte de 10–20% vs S0. La combinaison multiburst + variabilité de bruit crée des échantillons ambigus. C'est la meilleure approximation de la performance attendue sur données réelles.

### S4_real_noise — `--signal realistic --noise real`

**Question :** Le bruit réel du capteur dégrade-t-il au-delà du bruit synthétique ?

**Justification :** Utilise des segments de 2500 échantillons extraits des 305 fichiers de bruit réel acquis. Le bruit réel contient des composantes 1/f, des résonances d'amplificateur, et des artefacts de quantification que les modèles paramétriques ne capturent pas.

**Hypothèse :** Performance proche de S3. Si écart > 5%, le modèle de bruit realistic est insuffisant et le bruit réel contient des structures non modélisées.

---

## 3. Tier 2 — Ablation (isoler les facteurs)

### S5_signal_realism — `--signal realistic --noise none`

**Question :** Quel est l'impact des distorsions signal seules (DC offset, multiburst, skew) ?

**Justification :** En comparant S0 (pure/none) vs S5 (realistic/none), on mesure exactement la dégradation causée par les imperfections signal sans confusion avec le bruit. Cela quantifie la robustesse du modèle aux variations d'enveloppe et aux événements multi-particules.

**Hypothèse :** Perte de 3–8% vs S0. Le multiburst (10% des échantillons) devrait être le facteur dominant, car il modifie la structure temporelle du signal.

### S6_noise_realism — `--signal pure --noise realistic`

**Question :** Quel est l'impact du bruit réaliste seul (sans distorsions signal) ?

**Justification :** Symétrique de S5 : isole l'effet du bruit. Le preset realistic injecte le bruit *avant* le filtre passe-bande, ce qui est physiquement correct (le bruit capteur traverse le même filtre analogique que le signal).

**Hypothèse :** Perte de 5–12% vs S0. Le bruit est généralement le facteur dominant dans les systèmes OFI.

**Analyse d'additivité :** Si Δ(S5) + Δ(S6) ≈ Δ(S3) (où Δ = perte d'accuracy vs S0), les effets sont additifs. Si Δ(S3) > Δ(S5) + Δ(S6), il y a interaction (le bruit amplifie l'effet des distorsions signal). Si Δ(S3) < Δ(S5) + Δ(S6), il y a compensation (le modèle apprend des features plus robustes quand les deux perturbations sont présentes).

### S7_pure_real — `--signal pure --noise real`

**Question :** Le bruit réel dégrade-t-il différemment du bruit realistic sur un signal pur ?

**Justification :** Complète la grille d'ablation du bruit. En comparant S6 (pure/realistic) vs S7 (pure/real), on mesure le gap entre le modèle de bruit synthétique le plus fidèle et le vrai bruit capteur, en neutralisant l'axe signal.

**Hypothèse :** Écart < 3% entre S6 et S7. Si l'écart est plus grand, le modèle de bruit realistic sous-estime la complexité spectrale du bruit réel.

---

## 4. Tier 2b — Variation de SNR

### S8_colored_low — `--signal pure --noise colored` (σ=0.03)

**Question :** Comment la performance se dégrade-t-elle avec le SNR ?

**Justification :** Le σ=0.03 donne un SNR environ 2× meilleur que le preset standard (σ=0.058). Ce point de mesure, combiné avec S2 (σ=0.058) et S9 (σ=0.1), trace une courbe de dégradation en 3 points qui caractérise la sensibilité du modèle au niveau de bruit.

**Hypothèse :** Accuracy proche de S0 (< 2% d'écart). À faible bruit, le signal domine.

### S9_colored_high — `--signal pure --noise colored` (σ=0.1)

**Question :** Quelle est la limite de robustesse au bruit pour un signal pur ?

**Justification :** Le σ=0.1 donne un SNR environ 1.7× pire que le standard. C'est un stress test : le bruit commence à masquer les oscillations des petites particules (2 µm, dont le peak amplitude est ~0.17–0.23 mV).

**Hypothèse :** Perte de 8–15% vs S0, concentrée sur la classe 2 µm (SNR le plus faible). La courbe {S8, S2, S9} devrait montrer une dégradation non-linéaire avec un coude autour de σ=0.06–0.08.

---

## 5. Tier 3 — Validation réelle

### dataset (données expérimentales)

**Source :** 1511 fichiers acquis sur le capteur OFI réel (filenames `HFocusing_5_10_*.npy`).

**Rôle :** Ground truth pour la validation sim-to-real. Aucun modèle synthétique ne capture toutes les caractéristiques des données réelles (burst width 3–5× plus large, mismatch spectral 10 µm, bruit structuré, etc.). La performance sur le test set réel est la métrique finale du benchmark.

### S_union — concaténation de S0–S9

**Structure :** 12 090 fichiers train (4030 par classe, symlinks vers les 10 datasets synthétiques). Test set = real `dataset/test/` (302 fichiers).

**Question :** La diversité d'entraînement (domain randomization) améliore-t-elle le transfert sim-to-real ?

**Justification :** Chaque dataset synthétique expose le modèle à un sous-ensemble des perturbations réelles. L'union force le modèle à devenir invariant à toutes ces perturbations simultanément. En domain adaptation, la diversité des conditions d'entraînement est souvent plus importante que la fidélité d'une seule condition.

**Hypothèse :** S_union→real devrait surpasser tout S_i→real individuel. Si ce n'est pas le cas, certains datasets introduisent des biais contradictoires.

---

## 6. Protocole de cross-testing

### Matrice d'évaluation (11 × 12)

Chaque modèle est entraîné sur un dataset (lignes) et évalué sur tous les test sets (colonnes).

| Train ↓ / Test → | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Real | Self |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_baseline | — | . | . | . | . | . | . | . | . | . | . | ✓ |
| S1_white | . | — | . | . | . | . | . | . | . | . | . | ✓ |
| S2_colored | . | . | — | . | . | . | . | . | . | . | . | ✓ |
| S3_realistic | . | . | . | — | . | . | . | . | . | . | . | ✓ |
| S4_real_noise | . | . | . | . | — | . | . | . | . | . | . | ✓ |
| S5_signal_realism | . | . | . | . | . | — | . | . | . | . | . | ✓ |
| S6_noise_realism | . | . | . | . | . | . | — | . | . | . | . | ✓ |
| S7_pure_real | . | . | . | . | . | . | . | — | . | . | . | ✓ |
| S8_colored_low | . | . | . | . | . | . | . | . | — | . | . | ✓ |
| S9_colored_high | . | . | . | . | . | . | . | . | . | — | . | ✓ |
| S_union | . | . | . | . | . | . | . | . | . | . | . | ✓ |

Chaque cellule `.` contient l'accuracy du test croisé. La colonne **Real** est la plus importante : c'est le sim-to-real gap. La colonne **Self** est l'accuracy sur le propre test set (performance nominale).

---

## 7. Analyses comparatives clés

### A1 — Gradient de difficulté

**Comparaison :** Self-accuracy S0 > S1 > S2 > S3 > S4

**Ce qu'on apprend :** Le classement confirme (ou infirme) que la difficulté augmente avec le réalisme. Un renversement (ex: S3 > S2) indiquerait que le modèle bénéficie de certaines distorsions comme régularisation implicite.

**Métrique :** Self-accuracy par dataset, tracée en bar chart.

### A2 — Signal vs Bruit (ablation factorielle)

**Comparaison :**
- Δ_signal = acc(S0) − acc(S5) → impact des distorsions signal
- Δ_bruit = acc(S0) − acc(S6) → impact du bruit realistic
- Δ_combiné = acc(S0) − acc(S3) → impact combiné
- Test d'additivité : Δ_combiné vs Δ_signal + Δ_bruit

**Ce qu'on apprend :** Si Δ_combiné ≈ Δ_signal + Δ_bruit, les effets sont indépendants et chacun peut être traité séparément (ex: denoising pour le bruit, data augmentation pour le signal). Si Δ_combiné > somme, il y a synergie négative (le bruit amplifie l'impact des distorsions). Si Δ_combiné < somme, il y a un effet de régularisation croisée.

**Métrique :** Tableau Δ avec intervalles de confiance (multi-seed si possible).

### A3 — Bruit synthétique vs bruit réel

**Comparaisons :**
- S6 (pure/realistic) vs S7 (pure/real) → gap modèle de bruit, signal neutre
- S3 (realistic/realistic) vs S4 (realistic/real) → gap modèle de bruit, signal réaliste
- S2 (pure/colored) vs S7 (pure/real) → gap bruit coloré vs bruit réel

**Ce qu'on apprend :** Si S6 ≈ S7 et S3 ≈ S4, le modèle de bruit realistic est suffisant pour l'entraînement et il n'est pas nécessaire d'acquérir du bruit réel. Si S4 >> S3 (entraîné sur bruit réel surpasse bruit realistic), le modèle paramétrique est insuffisant.

**Métrique :** Accuracy diff + matrices de confusion comparées (quelles classes sont affectées ?).

### A4 — Courbe de sensibilité au SNR

**Comparaison :** S8 (σ=0.03) → S2 (σ=0.058) → S9 (σ=0.1)

**Ce qu'on apprend :** La courbe accuracy = f(σ) caractérise la robustesse au bruit. Le point d'inflexion (coude) indique le SNR critique en dessous duquel la classification se dégrade rapidement. Cela définit les spécifications minimales du capteur pour une classification fiable.

**Analyse par classe :** La dégradation sera non-uniforme. La classe 2 µm (amplitude la plus faible, m0 ∈ [7,14]) devrait être la première à se dégrader, suivie de 4 µm, puis 10 µm.

**Métrique :** Courbe acc(σ) globale + par classe. F1-score par classe pour identifier le point de rupture de chaque classe.

### A5 — Sim-to-real gap

**Comparaison :** Colonne "Real" de la matrice — accuracy de chaque modèle synthétique évalué sur le test set réel.

**Ce qu'on apprend :** Le dataset d'entraînement optimal pour le transfert sim-to-real. On s'attend à ce que S4 (realistic/real) donne le meilleur transfert, suivi de S3 (realistic/realistic). Si S0 (pur/sans bruit) transfert mieux que S3, cela signifie que les distorsions synthétiques ne correspondent pas à la réalité et ajoutent du bruit au lieu de régulariser.

**Métrique :** Accuracy sur real test set. Generalization gap = acc(self test) − acc(real test).

### A6 — Domain randomization (S_union)

**Comparaison :** S_union→real vs max(S_i→real) pour tout i ∈ {0..9}

**Ce qu'on apprend :** Si S_union gagne, la diversité est bénéfique : le modèle apprend des features invariantes aux conditions d'acquisition. Si S_union perd, l'ajout de données "trop faciles" (S0, S8) ou "trop différentes" dilue l'apprentissage.

**Extensions possibles :**
- Ablation S_union : retirer un dataset à la fois et mesurer l'impact (leave-one-out)
- Union sélective : ne garder que les top-K datasets pour le transfert

**Métrique :** Accuracy sur real test set. Comparer avec le meilleur dataset individuel.

### A7 — Transfert asymétrique

**Comparaison :** Pour chaque paire (i, j), comparer acc(train_i→test_j) vs acc(train_j→test_i).

**Ce qu'on apprend :** Le transfert est-il symétrique ? En général, un modèle entraîné sur des données "dures" (S3, S4) devrait transférer vers des données "faciles" (S0, S1), mais pas l'inverse. L'asymétrie quantifie la hiérarchie de difficulté et confirme le gradient du Tier 1.

**Métrique :** Heatmap de la matrice complète, annotée avec les asymétries significatives.

---

## 8. Hypothèses falsifiables (résumé)

| # | Hypothèse | Datasets | Critère de falsification |
|---|-----------|----------|--------------------------|
| H1 | Le gradient de difficulté suit l'ordre S0 > S1 > S2 > S3 ≥ S4 | S0–S4 | Tout renversement d'ordre |
| H2 | Le bruit est le facteur dominant (Δ_bruit > Δ_signal) | S0, S5, S6 | Δ_signal > Δ_bruit |
| H3 | Les effets signal et bruit sont approximativement additifs | S0, S3, S5, S6 | \|Δ_combiné − (Δ_signal + Δ_bruit)\| > 5% |
| H4 | Le bruit realistic est un bon proxy du bruit réel (écart < 3%) | S6 vs S7, S3 vs S4 | Écart > 5% |
| H5 | La dégradation avec σ est non-linéaire avec un coude | S8, S2, S9 | Dégradation linéaire |
| H6 | La classe 2 µm est la première à se dégrader avec le bruit | S8, S2, S9 | Autre classe dégrade d'abord |
| H7 | S_union transfert mieux que tout S_i individuel vers le réel | S0–S9, S_union | Un S_i individuel surpasse S_union |
| H8 | Le transfert est asymétrique (dur→facile meilleur que facile→dur) | Matrice complète | Symétrie observée |

---

## 9. Commandes de génération (référence)

```bash
# Tier 1 — Core
python generate_dataset.py auto --signal pure      --noise none      --output S0_baseline
python generate_dataset.py auto --signal pure      --noise white     --output S1_white
python generate_dataset.py auto --signal pure      --noise colored   --output S2_colored
python generate_dataset.py auto --signal realistic --noise realistic --output S3_realistic
python generate_dataset.py auto --signal realistic --noise real      --output S4_real_noise

# Tier 2 — Ablation
python generate_dataset.py auto --signal realistic --noise none      --output S5_signal_realism
python generate_dataset.py auto --signal pure      --noise realistic --output S6_noise_realism
python generate_dataset.py auto --signal pure      --noise real      --output S7_pure_real

# Tier 2b — SNR
python generate_dataset.py auto --signal pure --noise colored --config configs/S8_colored_low.ini  --output S8_colored_low
python generate_dataset.py auto --signal pure --noise colored --config configs/S9_colored_high.ini --output S9_colored_high

# Tier 3 — Union (domain randomization)
# S_union est créé par symlinks depuis les train sets de S0–S9, test set = dataset réel
```

---

## 10. Résumé des datasets

| ID | Nom | Signal | Bruit | σ | Train | Test | Tier |
|----|-----|--------|-------|---|-------|------|------|
| S0 | S0_baseline | pure | none | — | 1209 | 302 | Core |
| S1 | S1_white | pure | white | 0.058 | 1209 | 302 | Core |
| S2 | S2_colored | pure | colored | 0.058 | 1209 | 302 | Core |
| S3 | S3_realistic | realistic | realistic | 0.21* | 1209 | 302 | Core |
| S4 | S4_real_noise | realistic | real | 0.058 | 1209 | 302 | Core |
| S5 | S5_signal_realism | realistic | none | — | 1209 | 302 | Ablation |
| S6 | S6_noise_realism | pure | realistic | 0.21* | 1209 | 302 | Ablation |
| S7 | S7_pure_real | pure | real | 0.058 | 1209 | 302 | Ablation |
| S8 | S8_colored_low | pure | colored | 0.030 | 1209 | 302 | SNR |
| S9 | S9_colored_high | pure | colored | 0.100 | 1209 | 302 | SNR |
| — | S_union | mix | mix | mix | 12090 | 302† | Union |
| — | dataset | réel | réel | — | 1209 | 302 | Réel |

\* σ_base avant filtrage passe-bande (σ effectif post-filtre ≈ 0.058)
† Test set = données réelles (dataset/test)
