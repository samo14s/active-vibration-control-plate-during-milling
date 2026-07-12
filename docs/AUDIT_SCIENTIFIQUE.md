# Audit scientifique du package de simulation

**Thèse** : *Contribution au contrôle actif des vibrations en fraisage des pièces flexibles* (étude théorique, sans validation expérimentale)
**Périmètre** : l'intégralité du package (`01_core`, `02_controllers`, `03_analysis`, `04_figures`, `05_main`) — code, README et résultats annoncés.
**Date** : juillet 2026.

> ## ✅ STATUT : plan P0 EXÉCUTÉ (juillet 2026)
>
> Les 9 corrections P0 (§6) ont été appliquées et re-exécutées. Résultats
> honnêtes après correction (à comparer aux chiffres pré-correction cités
> dans le corps de l'audit) :
>
> | Grandeur | Avant (gonflé/faux) | Après (honnête) |
> |---|---:|---:|
> | Gain DARC moyen (S1-S4) | +55.2 % | **+53.1 %** |
> | S4 (K_T +30 % inconnu du contrôleur) | +52.6 % (oracle) | **+41.9 %** |
> | SLD LQG à 4900 RPM | 2.86 mm (substitution de pôles) | **2.375 mm** (Floquet BF complet) |
> | SLD « DARC » | 4.00 mm / « 40× » (boost 1.30× + plafond de grille) | **= SLD LQG** (théorème : FF exogène) |
> | Piézo réaliste (LQG) | −589 % (bug off-by-one retard) | **98.98 %** de réduction |
> | Gain NN hors-échantillon (trajet complet) | annoncé +15-25 % | **+4 points** (0.286 → 0.266 µm) |
> | λ Von Kármán | abs() d'une projection au signe faux | forme énergétique variationnelle |
> | « DARC-MPC » | aucun MPC dans le code | renommé **DARC**, code mort supprimé |
> | Baseline modèle interne (A6) | absente | **IMC-LQG** implémentée : 0.223 µm en S1
>   (bat DARC nominalement, sans modèle de coupe) mais diverge sous détuning −15 % |
>
> Les constats P1 (§6) restent ouverts et sont listés dans le README racine.

## Méthodologie

- Ré-exécution complète et indépendante de `main_simulation.py` sur un environnement propre (Python 3.11, NumPy 2.4, SciPy 1.17).
- Revue scientifique parallèle de 7 groupes de modules par des relecteurs indépendants (dynamique de l'usinage, mécanique des structures, piézoélectricité, automatique), chaque constat étant ensuite soumis à une **vérification adverse** (deux contre-expertises indépendantes par constat, sous les angles « exactitude mathématique » et « plausibilité physique/littérature ») avec ré-exécution numérique du code à l'appui.
- Bilan : **31 constats confirmés** (14 critiques, 17 majeurs), 3 constats **réfutés** après contre-expertise, plus une série de points mineurs.

---

## 1. Ce qui est vérifié et solide

Ces points survivront à une revue de jury ou de revue (MSSP/JSV/TCST) :

1. **Reproductibilité bit-exacte.** La ré-exécution indépendante reproduit *tous* les tableaux du README au chiffre imprimé près : S1–S4 (LQG 0.6052/1.2057/0.9234/0.7881 µm ; DARC 0.2954/0.5710/0.3383/0.3735 µm ; moyenne +55.19 %), le balayage bruit capteur (0.604/0.293 → 1.022/0.869 µm), le trajet complet (0.471/0.286/0.267 µm), et les profondeurs critiques SLD (0.100/2.862/4.000 mm). Graines aléatoires fixées partout, sensibilité aux graines négligeable.
2. **La référence pivot est réelle** : Nasiri & Moradi, *MSSP* 224 (2025) 112198 existe bien (ScienceDirect, S0888327024010975).
3. **Machinerie modale correcte** : fonctions de poutre encastrée-libre et libre-libre exactes à 10 chiffres, orthonormalité en masse vérifiée à la précision machine, quotient de Rayleigh avec l'énergie de Kirchhoff complète (terme de Poisson + torsion), couplage inter-modes de Ritz négligeable (< 5 %) — l'hypothèse diagonale est défendable.
4. **Couplage piézo classique et exact** : `B_piezo = −E_Pe·d31·(h_Pa+b_P)/(2(1−ν_Pe))` est la constante de moment de patch standard (théorie Dimitriadis/Fuller/Rogers) ; l'évaluation analytique par termes de bord de ∬∇²W coïncide avec la quadrature 2D à la précision machine ; autorité statique ~30 nm/V physiquement plausible.
5. **Les ancres FEM sont d'authentiques résultats FEM** : un assemblage indépendant de l'élément Q4 livré reproduit exactement [521.06, 1069.95, 2733.02] Hz sur un maillage 30×24, avec convergence en maillage propre.
6. **Intégration Newmark du terme régénératif linéaire algébriquement correcte** (terme courant implicite dans K_eff, terme retardé explicite, γ=1/2 sans amortissement numérique — les ζ = [0.31, 0.17, 0.27] % sont préservés exactement pour les modes 1–2).
7. **Bonnes pratiques de comparaison** : bruit capteur apparié (mêmes tirages pour LQG et DARC), pas de cherry-picking de métriques (RMS/pic/effort sur la même fenêtre pour les deux), la « base LQG identique » de `main_simulation.py` est vraie **au niveau du code** (matrices, poids, Kalman, ZOH identiques octet pour octet).
8. **L'ablation trois voies sur le trajet complet** (`main_fullpath_comparison.py` : LQG → +FF → +FF+NN sur 20.4 s, NN entraîné sur un segment de 0.5 s) est une expérience bien conçue : c'est la seule évaluation du NN sur données *non vues*, et le README rapporte honnêtement l'effondrement du gain NN (~20 % → ~7 %) hors segment.
9. **Le cœur théorique du feedforward est défendable** : aux harmoniques exactes de passage de dent, le terme régénératif s'annule pour la réponse τ-périodique (1−e^{−jωτ}=0), donc l'inversion du modèle en boucle fermée à ces harmoniques est un design légitime — le gain de ~39–40 % de la couche FF se reproduit sur un horizon 40× plus long que sa fenêtre de design.

---

## 2. Constats critiques (14)

### A. Le contrôleur proposé et l'équité de la comparaison

**A1. Il n'y a aucun MPC dans « DARC-MPC », et les composants « Adaptive » et « Robust » sont du code mort.** (`02_controllers/darc_mpc_v3_controller.py:680`)
Le pipeline `step()` est : Kalman → gain LQG → table périodique `u_ff[k mod 82]` → passe avant NN → `np.clip`. Aucun horizon de prédiction, aucune optimisation à horizon glissant, aucun QP sous contraintes. `OnlineRLSAdapter.update()` retourne une valeur jamais relue ; `omega_hat` ne modifie rien ; le filtre de Lyapunov est court-circuité dans toutes les expériences rapportées. De l'acronyme, seuls « Deep » (perceptron 16 neurones) et la base LQG existent fonctionnellement. **Nommer cela « DARC-MPC » devant un jury ou une revue sera traité comme une fausse représentation de la méthode.**

**A2. Contamination train/test : le NN est entraîné, et son meilleur checkpoint sélectionné, sur le scénario d'évaluation lui-même.** (`05_main/main_simulation.py:206`, `darc_mpc_v3_controller.py:631–658`)
`train_nn_residual` utilise les mêmes séquences α, la même trajectoire outil, le même T_end que l'évaluation notée ; pour S1/S2/S4, plante d'entraînement = plante de test. Pire : la sélection du meilleur réseau se fait **sur le RMS du scénario de test**. Le gain NN de +15–25 % par scénario est de la mémorisation in-sample ; le propre test hors-échantillon du package (trajet complet) le voit s'effondrer à ~+4 points. **L'avantage généralisable et défendable est le ~40 % du feedforward, pas 55 %.**

**A3. Le chiffre-titre de ~55 % est quasi garanti par construction : le feedforward reçoit une connaissance-oracle de la perturbation et de la plante.** (`05_main/main_simulation.py:203`)
`design_periodic_feedforward` reçoit le tableau α3 *identique octet pour octet* à celui qui génère la force dans le simulateur, calculé avec les paramètres *réels* du scénario ; le modèle inversé est le même espace d'état que celui intégré ; le verrouillage de phase est exact par construction (n_per = n_tau = 82). « Un feedforward avec modèle parfait de la perturbation et de la plante bat le feedback » est une tautologie de manuel. Aucune expérience ne teste une erreur de modèle de perturbation (K_T mal estimé, faux-rond, jitter de phase broche).

**A4. Dans S4 (« High K_T +30 % »), le feedforward et le NN sont (re)conçus avec les coefficients de coupe VRAIS perturbés.** (`main_simulation.py:169–171, 203, 206`)
Le test de robustesse donne au contrôleur proposé la perturbation qu'il est censé subir, pendant que le LQG (pur feedback) ne reçoit rien. Contre-factuel mesuré : avec un FF+NN conçus sur le K_T *nominal* (ce qu'un vrai contrôleur aurait), le gain S4 tombe de +52.6 % à **+41.6 %**. S3 (détuning fréquentiel), en revanche, est honnêtement conçu (FF/NN sur plante nominale, simulation sur plante détunée).

**A5. Le pipeline des figures de publication contredit la « base LQG identique » : il dé-règle délibérément le baseline.** (`04_figures/gen_article_complete_figures.py:190–210`)
Le générateur des « 14 figures principales » construit le LQG avec w_q=1e13, commenté *« SUB-OPTIMAL weights (typical engineer's guess) »*, tandis que DARC reçoit la base optimale w_q=1e14 (*« best of both worlds »*), avec capteur idéal et un pré-entraînement NN différent de celui du tableau README. **Figures et tableaux proviennent de deux expériences incohérentes, dont l'une est inéquitable de son propre aveu — ce commentaire, s'il est découvert, détruit à lui seul le récit de la comparaison équitable.**

**A6. Le baseline LQG est un homme de paille : l'observateur de Kalman n'a aucun modèle de la perturbation de coupe, alors que DARC reçoit la perturbation exacte.** (`02_controllers/lqg_controller.py:61–75`)
L'excitation dominante est une force périodique déterministe (245 Hz et harmoniques) ; l'observateur n'a ni canal d'entrée pour elle ni états de perturbation augmentés. Mesuré en S1 avec capteur idéal : erreurs d'estimation d'état de 50–120 % en RMS relatif. L'affirmation du README « un feedback ne peut pas rejeter une perturbation périodique » est **fausse en l'état** : avec la période de broche connue (l'encodeur est déjà supposé pour DARC), le principe du modèle interne (Francis & Wonham 1976), la commande répétitive (Hara et al. 1988) ou le LQG à accommodation de perturbation (Johnson 1971) rejettent asymptotiquement ces harmoniques par feedback. **Le concurrent naturel (LQG à Kalman augmenté d'harmoniques / commande répétitive) est absent de l'étude.**

### B. Physique du modèle

**B1. Coefficient cubique de Von Kármán : la projection de Galerkin donne un λ NÉGATIF (assouplissant) pour les 3 modes ; le signe est inversé en silence par `np.abs()`, et l'amplitude ne correspond à aucune dérivation cohérente.** (`01_core/plate_model.py:306–310`)
Projection brute : λ_brut = [−4.77e13, −9.12e14, −2.79e15] (signe parasite dû au flux de bord de la forme forte avec u=v=0 sur trois bords libres). La forme énergétique variationnellement cohérente (garantie positive) donne [1.06e14, 1.11e15, 1.42e16] — un facteur 2.2–5.1× plus grand que |λ| stocké. Le λ stocké n'est ni le résultat en forme forte (signe faux), ni le résultat énergétique u=v=0 (2–5× d'écart), ni le coefficient de l'article (qui résout le problème membranaire). **Toute affirmation « non linéaire » construite sur ce terme est indéfendable en l'état.**

**B2. L'amplitude de broutement en boucle ouverte est bornée par des boutons numériques artificiels, pas par la physique de séparation de l'article.** (`01_core/newmark_solver.py:116–117, 221–240`)
α4 < 0 pendant l'engagement, donc le terme cubique de coupe *amplifie* la croissance au lieu de la limiter ; la bornitude vient (a) du clip `chip_sat = 10·f_t` (bouton jamais réglé par aucun appelant) et (b) d'un seuil de séparation mal échelonné. Mesuré : cycle limite boucle ouverte à 1717 µm — **17× l'engagement radial de 0.1 mm** (l'outil perd physiquement le contact à |w| ~ a_e) et ~43 % de l'épaisseur de la plaque ; faire passer chip_sat de 10·f_t à 20·f_t déplace le « cycle limite » de 1717 à 1203 µm. Pas de régénération multiple (mémoire de surface à 2τ, 3τ) contrairement à l'article, §4.1.

### C. L'étude « actionneur réaliste »

**C1. Bug off-by-one dans le buffer de retard capteur : le retard réel est n_delay+1 échantillons (100 µs au lieu des 50 µs spécifiés).** (`01_core/piezo_actuator.py:148–156`)
Vérifié numériquement : avec `sensor_delay=50e-6` et dt=50 µs, la sortie est y[k−2] ; même `sensor_delay=0` produit 1 échantillon de retard. Ablation sur la configuration livrée : boucle stable et quasi idéale à 1 échantillon (max|y| = 2.58 µm), **catastrophiquement instable à 2 échantillons (max|y| = 1736 µm, 9017/10001 pas saturés)**. Correctif : dépiler quand `len(buffer) > n_delay`.

**C2. L'étude realistic-piezo livrée montre le LQG déstabilisant la plante (1736 µm, pire que la boucle ouverte, « réduction −589 % ») ; la marge de retard de la base LQG/DARC des résultats-titres est < 100 µs, alors que tous les résultats-titres supposent une boucle à latence nulle.** (`05_main/main_realistic_piezo.py`)
Tous les tableaux du README sont calculés avec `piezo=None` (zéro retard capteur, zéro dynamique d'actionneur). Une revendication de contrôle dont la stabilité meurt entre un et deux échantillons de latence supplémentaire (un capteur à courants de Foucault + DSP + ampli HT dépasse facilement 100 µs) n'est pas défendable en pratique — et le package, tel que livré, imprime lui-même la contre-preuve.

### D. SLD et robustesse

**D1. La courbe SLD de DARC est fabriquée par un multiplicateur d'amortissement 1.30× codé en dur ; un feedforward verrouillé en phase ne peut, de manière prouvable, pas déplacer les multiplicateurs de Floquet.** (`04_figures/gen_SLD_academic_style.py:142`, `main_simulation.py:673–676`)
u_FF(φ) est une entrée exogène périodique indépendante de l'état : dans une EDR linéaire périodique elle ne change que la solution particulière, pas l'équation variationnelle homogène — la matrice de monodromie est strictement inchangée. **Dans le cadre linéaire du package, le vrai SLD de DARC est identique à celui de sa base LQG.** La justification en commentaire cite des chiffres périmés (« réduction 4.7 % ») contradictoires avec les 51 % actuels ; de plus « a_p,crit = 4.00 mm » est exactement le plafond de la grille (`ap_arr[-1]`), pas un croisement calculé. Le « 40× » du tableau est un artefact de grille empilé sur un cadran heuristique.

**D2. Le SLD « multi-modes » est de facto mono-mode : la moyenne signée de la déformée le long du trajet annule exactement le mode 2 (torsion) et supprime le mode 3 d'un facteur ~1600 en couplage.** (`gen_SLD_academic_style.py:109–112`, `fdm_stability.py:113`)
Dp_avg = [6.63, 3.9e−16, 0.33] contre |Dp|_max = [6.63, 11.48, 13.26]. Au pire cas de position, le mode 2 a une limite de broutement **comparable ou inférieure** à celle du mode 1 (~0.055–0.085 mm) — totalement invisible dans le SLD livré. Le simulateur temporel applique pourtant le couplage complet dépendant de la position : le SLD analyse une autre plante que celle simulée.

**D3. L'analyse Monte Carlo de robustesse est du code mort.** (`03_analysis/uncertainty_analysis.py:63`)
`run_monte_carlo` n'est appelé par aucun script ; l'API documentée (`run_uncertainty_analysis`) n'existe pas ; les revendications du README (100 tirages, ±15 %, IC 95 %) ne correspondent ni aux défauts du code (50 tirages ; ±2 %/±5 %/±20 %) ni à aucun pipeline exécuté ; aucun intervalle de confiance n'est calculé nulle part ; la « Figure 14 : Robustness Monte Carlo » est un diagramme en barres déterministe sur 4 scénarios. À noter : le protocole *implémenté* (contrôleur conçu sur le nominal, simulé sur le perturbé) est le bon — il n'est simplement jamais exécuté.

---

## 3. Constats majeurs (17) — synthèse

| # | Module | Constat | Localisation |
|---|--------|---------|--------------|
| M1 | plate | La « calibration » écrase les fréquences des modes 2–3 de 8.2 %/10.2 % en gardant les déformées de la base inexacte → FRF hybride cohérente avec aucun des deux modèles (facteurs 0.90–0.99, pas « ≈1 ») | `plate_model.py:221–237` |
| M2 | plate | Fréquences analytiques documentées « 528/1165/2657 Hz » fausses : le code produit 528/1165/**3042** Hz (doc périmée, écart 14 %) | `plate_model.py:27`, README:27 |
| M3 | plate | `set_process_damping()` jamais appelé — l'amortissement de process est revendiqué comme fonctionnalité mais ζ_p = 0 dans tous les résultats ; Γ est un bouton libre non dérivé | `plate_model.py:390` |
| M4 | force | Mode 3 (2733 Hz) résolu par 7.3 échantillons/période : ~6.1 % d'élongation de période Newmark, aucune étude de convergence en dt ; le FF vise des harmoniques jusqu'à 7.35 kHz (2.7 éch./période) | `newmark_solver.py:146` |
| M5 | force | Le « polynôme d'épaisseur de copeau d'ordre 3 » est un re-scaling heuristique du modèle linéaire hérité (noyaux angulaires cos²/cos³ faux, coefficients radiaux δ de la Table 2 jamais utilisés, Taylor en h₀=0) | `milling_force.py:135–143` |
| M6 | force | Usure en dépouille (Éq. 39), séparation (Éq. 6) et amortissement de process (Éq. 44–46) : trois fonctionnalités annoncées, trois codes morts ; l'effort d'arête tel que câblé serait une force DC sans effet dynamique | `milling_force.py:165–196` |
| M7 | piezo | Le modèle « Bouc-Wen » est exactement un filtre linéaire (superposition vérifiée à 1.4e−14) : aucune hystérésis réelle — aucune revendication de robustesse à l'hystérésis n'est soutenue | `piezo_actuator.py:112–122` |
| M8 | piezo | DARC-MPC n'est **jamais** simulé avec l'actionneur réaliste, alors que son feedforward en boucle ouverte est le composant le plus exposé aux erreurs gain/phase d'actionneur | `main_realistic_piezo.py:25` |
| M9 | piezo | Discrétisation de l'ampli : le lag 5 kHz annoncé se comporte comme ~12 kHz (−2.44° au lieu de −5.95° à 521 Hz) — erreur de discrétisation de la taille de l'effet étudié | `piezo_actuator.py:47` |
| M10 | lqg | Le « filtre de Kalman » n'en est pas un : W=1e−6·I, V=1e−12 codés en dur, jamais accordés au bruit simulé (0.1–2 µm) ; aucune optimalité revendicable — c'est un observateur de Luenberger arbitraire | `lqg_controller.py:61–66` |
| M11 | lqg | Critère de « grid search » scientifiquement faux : maximise la décroissance du pôle le PLUS RAPIDE (min Re) au lieu de l'abscisse spectrale ; coût LQ jamais évalué ; `except: pass` avale les échecs de Riccati (coïncidence : le point retenu est le bon) | `lqg_controller.py:117–125` |
| M12 | lqg | ζ ≈ 28 % et « a_p,crit LQG = 2.86 mm (28.6×) » dérivés de eig(A−BK) plein état : sans observateur, sans échantillonnage, sans couplage au retard régénératif, sans limite ±150 V (extrapolation linéaire : la tension dépasse 150 V bien avant 2.86 mm) | `lqg_controller.py:116`, `main_simulation.py:661–667` |
| M13 | stab | Le SLD « LQG » est une substitution de pôles dans le modèle SISO boucle ouverte, pas une analyse de Floquet de la boucle fermée (observateur + compensateur numérique absents de la monodromie — augmentation pourtant faisable dans ce cadre) | `gen_SLD_academic_style.py:126–134` |
| M14 | stab | SLD interne inéquitable : courbe LQG avec base sous-optimale w_q=1e13, courbe DARC à partir de w_q=1e14 avant le boost 1.3× — en contradiction avec la bannière « base identique » | `gen_SLD_academic_style.py:125–142` |
| M15 | mains | Verrouillage de phase broche parfait par construction ; sensibilité mesurée : DARC devient **PIRE que le LQG** au-delà de ~30–40° d'erreur de phase (22° ≈ 0.25 ms sur la période de dent) ; aucun test de jitter/variation de vitesse | `darc_mpc_v3_controller.py:700` |
| M16 | mains | Le README du module 05 affirme que l'étude realistic-piezo montre « DARC-MPC remains effective (~+15 %) » et liste une dérive thermique : le script ne simule pas DARC et le modèle thermique n'existe pas | `05_main/README.md:100–115` |
| M17 | mains | L'entrée SLD « DARC 4.00 mm / 40× » repose sur le multiplicateur arbitraire 1.30× justifié par des chiffres périmés d'une version antérieure (trois générations de résultats coexistent : 4.7 %, +19 %, 51–63 %) | `main_simulation.py:670–676` |

## 4. Constats réfutés après contre-expertise (à ne PAS corriger)

1. *« Le cubique de Von Kármán est dynamiquement sans effet à toute amplitude atteignable »* — *réfuté* : l'arithmétique du constat était exacte mais sa conclusion dynamique est empiriquement fausse dans le régime de broutement à grande amplitude du code.
2. *« Retard quantifié à la grille : la simulation modélise 4878 RPM et non 4900 »* — *réfuté comme majeur* : le fait est exact (n_τ = round(81.633) = 82, soit −0.45 % de vitesse effective) mais la cohérence interne périodique est préservée ; à traiter comme point mineur de rédaction (annoter la vitesse effective).
3. *« L'affirmation "sensor-independent" est fausse car le NN consomme l'estimée pilotée par le capteur »* — *réfuté comme majeur* : mécaniquement exact (le NN reçoit x̂ au pas 702), mais l'inférence centrale ne tient pas ; à réduire à une correction de formulation dans le README.

## 5. Points mineurs notables

- ρ = 2830 kg/m³ n'est pas de l'AL6061 (2700 kg/m³) — c'est une densité de classe 2024/7075 ; l'erreur (+2.4 % sur les fréquences si corrigée) dépasse la « calibration » du mode 1. Relabelliser l'alliage ou re-simuler.
- Le patch piézo n'ajoute ni masse ni raideur au modèle (≈7 % de la masse de la plaque, ignorés) — à quantifier/assumer.
- Le cubique de VK est diagonal-seulement (couplages inter-modes λ_ijkl abandonnés sans le dire).
- Élément Q4 (ACM) non conforme jamais signalé ; convergence vérifiée ici, mais la table de convergence doit figurer dans la thèse.
- Amplitudes FFT sous-estimées ~2× (gain cohérent de la fenêtre de Hann non compensé) sur des tracés étiquetés en µm.
- `sensor_floor` est une zone morte dure (sortie exactement nulle sous le seuil), pas un modèle de résolution physique.
- Divers écarts doc/code : « LQG y_RMS ≈ 0.63 µm » vs 0.605 réel ; README du module 05 documentant encore le pipeline FEM obsolète ; m_div 30 vs 40 selon les scripts.

---

## 6. Plan d'action priorisé

### P0 — indispensable avant toute soumission ou défense

1. **Renommer le contrôleur** (il n'y a pas de MPC) — p. ex. « LQG + feedforward inverse-modèle synchronisé broche + résidu neuronal ILC » — et supprimer ou implémenter réellement les couches Adaptive/Robust/Lyapunov mortes (A1).
2. **Protocole d'apprentissage propre** : entraîner le NN sur des épisodes disjoints de l'évaluation (autre segment, autre réalisation de bruit, plante nominale seulement), sélection de checkpoint sur un ensemble de validation, jamais sur le test (A2).
3. **Refaire S4 honnêtement** : FF+NN conçus sur K_T nominal, plante à 1.3·K_T (résultat attendu ≈ +42 %, qui reste bon) ; ajouter un scénario d'erreur de modèle de force pour le FF (A3, A4).
4. **Unifier le pipeline de figures avec `main_simulation.py`** : même base LQG optimale pour les deux contrôleurs partout ; purger le commentaire « sub-optimal / best of both worlds » (A5).
5. **Corriger l'off-by-one du retard capteur** (`piezo_actuator.py` : dépiler quand `len(buffer) > n_delay`) et ré-exécuter l'étude realistic-piezo ; documenter la marge de retard de la boucle (C1, C2).
6. **Von Kármán** : remplacer la projection en forme forte + `np.abs()` par la forme énergétique variationnelle (positive par construction), ou par la résolution membranaire de l'article ; sinon, retirer la couche « non linéaire » des revendications (B1).
7. **SLD honnête** : supprimer la courbe « DARC » (théoriquement identique à celle du LQG dans ce cadre linéaire) ou calculer un vrai Floquet de la boucle fermée augmentée ; retirer « 4.00 mm / 40× » (plafond de grille) ; recalculer la courbe LQG avec compensateur dans la monodromie (D1, M13, M14).
8. **Purger toutes les revendications de code mort** : process damping, usure, séparation Éq. 6, Monte Carlo « 100 tirages/IC 95 % », dérive thermique, « +15 % DARC sous piézo réaliste » (M3, M6, D3, M16).
9. **Ajouter le baseline feedback à modèle interne** (LQG à Kalman augmenté des harmoniques de dent, ou commande répétitive) — c'est LA question que posera tout rapporteur d'automatique (A6).

### P1 — pour consolider la thèse

10. Balayage d'**erreur de phase broche / jitter / dérive RPM** symétrique au balayage bruit capteur (M15).
11. **Simuler DARC sous l'actionneur réaliste corrigé** (M8) — après le correctif C1.
12. **Étude de convergence en dt** (50 → 10 µs) : tableau RMS et positions des pics FFT (M4).
13. **SLD multi-modes au pire cas de position** (Dp au max le long du trajet, pas la moyenne signée) (D2).
14. **Exécuter réellement le Monte Carlo** (le protocole implémenté est le bon), rapporter le taux d'instabilité et de vrais IC (D3).
15. **Recalibrer le Kalman** par niveau de bruit dans le balayage fig10 ; dériver W du canal de force (M10).
16. Étendre l'étude au **fraisage en avalant ET en opposition**, et documenter la vitesse effective 4878 RPM.

### P2 — qualité éditoriale

17. Corriger ρ/alliage, fréquences documentées (3042 Hz), gain de Hann, docs de modules périmées, dériver Γ si l'amortissement de process est conservé.

---

## 7. Ce qui reste défendable — recadrage suggéré de la contribution

Après corrections P0, la contribution honnête et solide de la thèse est :

1. **Un cadre de simulation reproductible bit-exact** du fraisage d'une plaque flexible instrumentée piézo (modèle modal analytique calibré FEM + NDDE régénérative + intégrateur Newmark cohérent) — la reproductibilité exemplaire est un vrai atout pour une thèse théorique.
2. **Un feedforward périodique inverse-modèle synchronisé broche apportant ~+39–40 % de réduction RMS sous base feedback identique**, robuste au bruit capteur et validé sur un horizon 40× la fenêtre de design — résultat réel *dans le modèle*, à borner par les études de sensibilité (phase, K_T, actionneur).
3. **Une couche résiduelle neuronale ILC** dont le gain hors-échantillon honnête est ~+4–7 points — modeste mais réel, à présenter comme tel.
4. **Une analyse critique des limites** (marge de latence < 100 µs de la base LQG, sensibilité du FF à la phase broche, domaine de validité linéaire) — transformer les faiblesses découvertes en chapitre de discussion est exactement ce qu'un jury attend d'une thèse purement théorique.

**Verdict global** : l'ossature linéaire (plaque modale, couplage piézo, force régénérative linéaire, intégrateur, base LQG) est saine et reproductible ; les couches « non linéaire », « MPC/adaptatif/robuste », le SLD contrôlé et la moitié des revendications de robustesse ne survivraient pas à une relecture hostile en l'état. Le package est un **excellent point de départ** à condition d'exécuter le plan P0 avant toute communication scientifique.
