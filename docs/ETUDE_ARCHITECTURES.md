# Étude d'architectures de commande — la structure d'information et sa contrepartie de robustesse

**Thèse** : *Contribution au contrôle actif des vibrations en fraisage des pièces flexibles*
**Objet** : comparer, sur un banc de simulation unifié et honnête (mêmes base
LQG, mêmes scénarios, protocole sans oracle), quatre architectures de rejet de
la perturbation régénérative/de passage de dent, classées par la **structure
d'information** qu'elles exploitent — et en tirer une conclusion **fondamentale**
sur le compromis performance ↔ robustesse.

Reproductible par : `python 05_main/main_imc_baseline.py` (≈ 40 s).

---

## 1. Les quatre architectures et leur structure d'information

| Contrôleur | Information exploitée | Mécanisme de rejet | Fichier |
|---|---|---|---|
| **LQG** | capteur de déplacement | retour optimal (aucun modèle interne du forçage) | `lqg_controller.py` |
| **IMC-LQG** | capteur + **période broche** (encodeur) | modèle interne harmonique (Kalman augmenté), gains d'annulation issus du modèle NOMINAL | `imc_lqg_controller.py` |
| **DARC-FF** | encodeur + **modèle de coupe** (a₃ nominal) | feedforward inverse-modèle pré-calculé, verrouillé en phase | `darc_controller.py` |
| **STSMC+DOB** | capteur + encodeur, **SANS modèle de coupe** | super-twisting (2e ordre) + observateur de perturbation intégral | `smc_dob_controller.py` |

---

## 2. Résultats (protocole honnête — aucun contrôleur ne reçoit les paramètres réels perturbés)

`y_RMS` (µm), T = 0.5 s, capteur fin 0.1 µm :

| Scénario | LQG | IMC-LQG | DARC-FF | STSMC+DOB |
|---|---:|---:|---:|---:|
| S1 — nominal | 0.605 | **0.223** | 0.363 | 0.608 |
| S2 — a_p = 0.6 mm | 1.206 | **0.475** | 0.728 | 1.207 |
| S3 — détuning ω −15 % | 0.923 | **15.76 ✗** | **0.592** | 0.917 |
| S4 — K_T +30 % inconnu | 0.788 | **0.293** | 0.536 | 0.790 |

(✗ = divergence : saturation ±150 V, instabilité.)

---

## 3. La fragilité DIRECTIONNELLE — chaque architecture casse dans un sens différent

- **IMC-LQG** : meilleure performance nominale (0.223 µm) et sous K_T inconnu
  (0.293 µm — le modèle interne s'adapte en ligne, **sans modèle de coupe**),
  MAIS **diverge sous détuning structurel S3** : ses gains d'annulation
  γ_h = −G_wy(jω_h)/G_uy(jω_h), calculés sur le modèle nominal, sont
  **déphasés dans la région résonante** quand la structure se désaccorde de
  15 % (le déphasage de G_uy est brutal près d'un mode faiblement amorti) →
  le « feedforward » injecte de l'énergie à contre-phase.
- **DARC-FF** : le seul qui **améliore S3** (0.592 µm : son feedforward est
  verrouillé sur la PHASE broche, insensible au détuning structurel), MAIS il
  **paie l'erreur de modèle de coupe** (S4 : 0.536 µm avec un feedforward conçu
  sur K_T nominal ; un feedforward « oracle » ferait mieux).
- **STSMC+DOB** : le **coin robuste** — récupère le LQG partout (0.6–1.2 µm),
  **sans modèle de coupe**, **sans jamais diverger** (S3 : 0.917 µm, là où IMC
  explose), avec garantie de convergence en temps fini. MAIS il **ne dépasse
  pas le LQG** : il n'achète pas la performance de pointe.

Aucune architecture ne domine : **acheter la performance (IMC/DARC) introduit
une fragilité ; garantir la robustesse (STSMC+DOB) coûte la performance.**

---

## 4. Conclusion FONDAMENTALE — pourquoi le coin robuste ne peut pas battre le LQG

Ce n'est pas un défaut de réglage. Une exploration systématique (5 variantes de
SMC/DOB : ISM super-twisting, super-twisting sur la sortie, NDOB statique, AFC
adaptatif verrouillé sur l'encodeur, NDOB + démodulation + gains inverse-modèle)
donne le même verdict, pour une raison STRUCTURELLE :

> **Sur une pièce faiblement amortie (ici ζ = 0.17–0.31 %), la 2ᵉ harmonique de
> passage de dent (2 × 245 = 490 Hz) tombe SUR le mode 1 (521 Hz).**
> Il en résulte un **conflit** entre :
> - **(a) l'estimation/annulation ROBUSTE EN LIGNE** de la perturbation (DOB,
>   AFC, modèle interne adaptatif) — toute inversion de phase en ligne est
>   **instable dans la région résonante** (le gain 1/|G_uy| y explose et la
>   phase y bascule de 180°) ;
> - **(b) l'annulation SÉLECTIVE en fréquence** qui, seule, permet de descendre
>   sous le LQG (c'est ce que font IMC et DARC).
>
> Un contrôle robuste à **large bande** (SMC/DOB) ne peut donc PAS rejeter
> sélectivement la perturbation périodique quasi-résonante : il **récupère
> l'optimum du retour** (LQG). L'annulation sélective (IMC/DARC) exige un
> **modèle** (calculé hors ligne, donc non robuste) et **paie sa fragilité**.

C'est la forme **rigoureuse, quasi-« impossibilité »**, du compromis de
structure d'information : *sur ce plateau, on ne peut avoir simultanément le
rejet périodique sous-µm ET la robustesse en ligne dans les deux directions.*

**Portée / limite** : ce verdict est spécifique au régime « harmonique de dent
sur résonance + très faible amortissement ». Sur une pièce plus amortie, ou à
une vitesse de broche plaçant les harmoniques loin des résonances (ex. le modèle
poutre-Timoshenko plus amorti de l'étude ADRC-FOPID de l'auteur, où un ESO/DOB
obtient 68.5 % de réduction), les méthodes DOB peuvent redevenir efficaces —
c'est une prédiction testable et une piste de travail (§6).

---

## 5. Ce que le STSMC+DOB apporte réellement (positionnement honnête)

À présenter comme le **contrôleur de robustesse garantie**, pas de performance :

1. **Ne diverge jamais** — contrairement à IMC-LQG (S3), il reste stable sous
   toute la plage d'incertitude testée.
2. **Sans modèle de coupe** — contrairement à DARC-FF, aucun K_T, aucun a₃
   requis (l'observateur de perturbation intégral du super-twisting est
   model-free).
3. **Convergence temps fini + certificat de robustesse Lyapunov** à
   l'incertitude appariée — garantie théorique que le LQG (asymptotique) et
   l'ADRC-FOPID (ESO asymptotique) n'offrent pas.
4. **Effort de commande faible** (16 V, comme le LQG) — pas de sur-actionnement.

Sa contrepartie assumée : il ne descend pas sous le LQG sur la vibration forcée.

---

## 6. AIMC — la résolution du conflit par l'adaptation INDIRECTE (par identification)

Le §4 établit que l'adaptation DIRECTE des coefficients d'annulation
(AFC/FxLMS) est instable en phase près de la résonance, et que les gains
FIGÉS (IMC/DARC) paient leur fragilité. La troisième voie — celle qui comble
la lacune de la littérature (« l'identification en cours d'usinage alimente la
re-planification des paramètres, jamais le chemin d'annulation actif ; le
FxLMS adapte des coefficients sur une dynamique FIXE ») — est l'adaptation
**INDIRECTE** : identifier LE MODÈLE en ligne, et RE-SYNTHÉTISER
algébriquement les gains d'annulation modèle-basés.

**AIMC** (`aimc_controller.py`, étude `main_aimc_study.py`) :
banc MMAE de filtres de Kalman augmentés sur une grille de détuning
(−20 %…+10 %), vraisemblances récursives avec oubli (fenêtre ≈ 10 ms),
**mélange bayésien** de l'annulation (u_canc = Σ p_i·Σ_h Re{γ̄_h(ρ_i)·ẑ_h^i},
aucun seuil, aucune commutation) au-dessus du retour LQG nominal fixe.
La boucle d'identification (stable par construction) est séparée de la boucle
de commande : aucun gradient n'est adapté dans la région résonante.

Résultats (protocole honnête, T = 0.5 s, et dérive T = 2 s) :

| Scénario | LQG | IMC figé | DARC-FF | STSMC+DOB | **AIMC** |
|---|---:|---:|---:|---:|---:|
| S1 nominal | 0.605 | 0.223 | 0.363 | 0.608 | **0.222** |
| S2 a_p=0.6 mm | 1.206 | 0.475 | 0.728 | 1.207 | **0.473** |
| S3 ω−15 % | 0.923 | 15.76 ✗ | 0.592 | 0.917 | **0.297** |
| S3b ω−12 % (hors grille) | 0.694 | 0.378 | 0.429 | 0.685 | **0.277** |
| S4 K_T+30 % inconnu | 0.788 | 0.293 | 0.536 | 0.790 | **0.292** |
| Dérive 0→−15 % en coupe, segment final | 0.631 | 8.49 (130 V) | 0.364 | 0.624 | **0.205** |

**L'AIMC domine (ou co-domine) TOUTE la carte** : il égale l'IMC figé là où
celui-ci excelle (S1/S2/S4), répare sa divergence sous détuning (S3 : 0.297 au
lieu de 15.8), interpole entre les points de grille (S3b), et SUIT une dérive
de fréquences en cours d'usinage là où l'IMC figé se dégrade de 40× (E3).
Le coût : ~0 % au nominal (0.222 vs 0.223) — l'adaptation est "gratuite" une
fois la vraisemblance concentrée.

Note d'identification : le ρ̂ identifié porte un biais systématique ≈ −3…−5 %
par rapport au détuning STRUCTUREL vrai — il identifie la fréquence EFFECTIVE
EN COUPE (raideur régénérative + couplage de retard inclus), qui est
précisément la bonne grandeur pour phaser l'annulation. (Cohérent avec le fait
connu que la FRF en coupe diffère de la FRF au marteau.)

Limites honnêtes : grille 1-paramètre (détuning global ρ — les modes réels
peuvent dériver indépendamment : grille multi-ρ = extension directe) ;
36 états de filtre au total (7×(6+16)) à 20 kHz — réaliste pour un DSP
moderne mais à chiffrer ; la dérive simulée est un proxy k_scale(t) uniforme.

## 6bis. Résultats étendus de l'AIMC (pour l'article — `main_aimc_extended.py`, `main_aimc_drift.py`)

Caractérisation complète, y compris les LIMITES (protocole honnête).

**E4 — paysage de détuning continu (−22 %…+12 %, au-delà de la grille MMAE).**
L'AIMC reste plat à 0.24–0.41 µm sur TOUT le continuum, tandis que les modèles
FIGÉS (LQG, IMC, DARC, STSMC) perdent la STABILITÉ dans les bandes de détuning
où un mode traverse un lobe (à ρ ≤ −16 %, le plateau nominal chatter à
a_p = 0.3 mm ; amplitudes non quantitatives car bornées par le clamp, mais la
DIVERGENCE l'est). L'AIMC identifie le plateau détuné et **maintient la
stabilité ET la performance sous-µm là où les autres décrochent** — y compris
juste HORS grille (ρ = −22 %, la grille s'arrête à −20 %), preuve d'un bord
gracieux.

**E5 — robustesse au bruit capteur (paroi détunée −15 %).** L'AIMC est de loin
le plus robuste : 0.35 µm (idéal) → 1.20 µm (2 µm RMS), contre 0.9→3.0 pour
LQG/DARC/STSMC et ~9 µm (déjà divergé) pour l'IMC figé. L'identification MMAE
survit au bruit car les vraisemblances intègrent sur ~10 ms.

**E7 — transitoire d'identification (entrée d'outil dans une paroi déjà
détunée −15 %).** Verrouillage à 90 % en **3 ms** ; RMS transitoire [0,50] ms
= 0.61 µm, établi = 0.27 µm. L'adaptation est quasi instantanée à l'échelle
d'une passe.

**E9 — annulation spectrale (paroi −15 %).** Atténuation des raies de passage
de dent : **−19 dB à 245 Hz, −11 dB à 490 Hz, −23 dB à 735 Hz, −16 dB à
980 Hz** (LQG → AIMC). Preuve fréquentielle directe du rejet harmonique.

**E10 — coût temps réel.** Banc VECTORISÉ (empilement des N filtres, une
`einsum` par pas) : **45 µs/pas < dt = 50 µs -> temps réel VÉRIFIÉ** (à 20 kHz).
Référence : LQG 7 µs, IMC 22 µs, DARC 15 µs, STSMC 15 µs. L'AIMC est ~2× l'IMC
(7 filtres augmentés = 7×22 = 154 états), et reste dans le budget. (La première
version, boucle Python, faisait 139 µs > dt ; la vectorisation était donc
nécessaire — la charge algorithmique elle-même est modérée pour un DSP.)

**E8 — bande passante de suivi (limite honnête).** La fenêtre d'oubli de 10 ms
donne une large bande de suivi : l'AIMC reste à **0.19–0.25 µm pour TOUS les
taux de dérive testés (8 %/s → 300 %/s)**, avec un retard d'identification
< 4 %. L'IMC figé, lui, diverge (8.7 → 15.8 µm) dès que le détuning s'accumule,
à toute vitesse. (Un enlèvement de matière réaliste dérive de quelques %/s ;
l'AIMC a donc une très large marge.)

**E12 — erreur d'encodeur / vitesse broche (limite honnête).** L'AIMC et l'IMC
partagent la sensibilité des modèles internes à la fréquence de dent supposée :
0.22 µm à erreur nulle → ~0.61 µm à ±4 % d'erreur de f_fund (l'AIMC reste
légèrement meilleur que l'IMC car il ré-identifie l'amplitude, pas la
fréquence). Le DARC, indexé par la PHASE (pas par f_fund), y est immunisé
(0.363 µm plat). **Conséquence de conception** : l'AIMC exige un bon
verrouillage sur l'encodeur (±4 % de RPM est énorme ; les encodeurs réels sont
< 0.1 %) — une grille MMAE en fréquence, ou l'ajout de la fréquence à
l'identification, lèverait cette sensibilité (P1).

## 7. Pistes de travail (P1 — recherche)

1. **Tester le STSMC+DOB sur une pièce plus amortie / poutre Timoshenko** (là
   où les DOB sont efficaces) : la prédiction du §4 est que l'avantage
   réapparaît hors du régime résonant.
2. **Balayage de vitesse de broche** : à un RPM plaçant les harmoniques de dent
   entre les modes (hors résonance), tester si l'annulation en ligne redevient
   stable et performante.
3. **Grille MMAE multi-paramètre** (ρ par mode, ± K_T) et grille adaptative
   (raffinement local autour du MAP).
4. **SLD de l'AIMC** : Floquet du système commuté/mélangé (LPV) — la frontière
   de stabilité d'un contrôleur à modèle identifié est un problème ouvert.
5. **Hybride DARC×STSMC** : feedforward inverse-modèle + dorsale
   super-twisting — le meilleur des deux, à valider.

---

## 8. Référence de positionnement (littérature, via Consensus)

- Le champ du contrôle actif de chatter des pièces minces est **essentiellement
  mono-architecture (feedback)** : modal actif, retour retardé optimal, H∞,
  MPC+Kalman, SMC+actionneur EM (Wan 2020), etc. **Aucun** ne formule le
  problème comme un compromis de **structure d'information**.
- Le **principe du modèle interne / la commande répétitive** est mûr en
  micatronique (moteurs, run-out broche, plateformes) mais **quasi absent** du
  fraisage de pièces flexibles — d'où l'originalité du baseline IMC-LQG et de la
  carte.
- **SMC+DOB** est mûr HORS fraisage (moteurs, robots, positionneurs) ; la
  contribution ici n'est pas l'algorithme mais **(i)** son adaptation au
  fraisage faiblement amorti, **(ii)** le constat fondamental du §4, **(iii)**
  la carte d'architectures unifiée — distinct de l'ADRC-FOPID (ESO+FOPID,
  asymptotique) de l'auteur.
