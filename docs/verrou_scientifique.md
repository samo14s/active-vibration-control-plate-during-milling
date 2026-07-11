# Verrou scientifique et positionnement de la contribution

*Document rédigé pour insertion dans le manuscrit de thèse*
**« Contribution au contrôle actif des vibrations en fraisage des pièces flexibles »**

Contribution mise en œuvre dans ce dépôt :
`02_controllers/darc_mpc_v4_plad_controller.py` (**PLAD** — *Phase-Locked
Adaptive DARC*), démontrée par `05_main/main_gap_spindle_sync.py` et
validée par `03_analysis/validate_phase_observer.py`.

---

## 1. Positionnement dans la thèse

Le titre de la thèse circonscrit trois éléments, tous présents dans ce
travail :

| Élément du titre | Réalisation dans le travail |
|---|---|
| **Contrôle actif des vibrations** | Actionneur piézoélectrique (patch QDA60-200.7) + observateur de Kalman + LQG + compensation anticipative |
| **en fraisage** | Fraisage périphérique, fraise 3 dents hélicoïdale, effort de coupe régénératif avec retard τ = 60/(N_T·N) |
| **des pièces flexibles** | Plaque mince AL6061 encastrée (cantilever, 100×80×4 mm) — cas canonique de paroi mince flexible |

La contribution PLAD n'est donc pas un sujet connexe : c'est une
**contribution à part entière au même axe**, qui ajoute une couche de
synchronisation robuste au contrôle actif déjà en place. Elle s'articule
naturellement comme **contribution finale** de la thèse, selon la
progression :

- **Contribution 1** — Modélisation (FEM Kirchhoff Q4 + réduction modale
  + modèle d'effort de coupe) et commande de référence **LQG**
  piézoélectrique.
- **Contribution 2** — Compensation anticipative apprise
  (**DARC-MPC**) qui dépasse le LQG en conditions nominales.
- **Contribution 3 (ce document)** — **Synchronisation sans capteur,
  robuste à l'incertitude de vitesse de broche (PLAD)** : la limite de la
  Contribution 2 est d'abord *démontrée* (elle s'effondre sous déviation
  de vitesse), puis *levée*.

Titre de chapitre suggéré :
**« Synchronisation sans capteur de la compensation anticipative pour la
robustesse à la variation de vitesse de broche »**.

---

## 2. Verrou scientifique

Le contrôleur de référence de la Contribution 2 (DARC-MPC v3) calcule

```
u(t) = u_LQG(x̂) + α · NN_FF(φ_horloge, x̂),   φ_horloge = 2π (k mod n_per)/n_per
```

Deux faiblesses structurelles en découlent directement (vérifiables dans
le code, `darc_mpc_v3_controller.py`) :

1. **Synchronisation en boucle ouverte.** La compensation anticipative
   apprise est indexée par le compteur de pas `k mod n_per` — une horloge
   en boucle ouverte qui suppose la période de passage de dent
   *exactement* connue et constante. Toute déviation réelle de la vitesse
   de broche désynchronise la compensation, alors injectée à une **phase
   erronée** qui dérive lentement. Le bénéfice anticipatif ne disparaît
   pas seulement : l'injection peut **dégrader** la réponse **sous** la
   ligne de base LQG.

2. **Adaptation non connectée.** Le v3 calcule à chaque pas un facteur de
   robustesse `lambda_robust` qui n'est **jamais appliqué** à la loi de
   commande : la couche « adaptative robuste » annoncée était inerte.

> **Énoncé du verrou.** Les stratégies de compensation anticipative
> (*feedforward*) apprise ou répétitive pour le contrôle actif des
> vibrations en fraisage sont synchronisées par une **horloge en boucle
> ouverte** ou par un **déclencheur codeur**, sous l'hypothèse d'une
> vitesse de broche exactement connue et constante. Des remèdes robustes
> à l'erreur de période existent dans la littérature générale de la
> commande, mais ils exigent **soit un codeur de broche**, soit
> sacrifient la performance de rejet périodique, et **aucun n'a été
> appliqué au contrôle actif de la rerforation (chatter) des pièces
> flexibles à parois minces**. La **synchronisation sans capteur**
> (sans codeur), à partir du **signal vibratoire lui-même**, d'une
> **forme d'onde de compensation apprise**, assortie d'un **repli
> sécurisé par gating de confiance** vers la commande par rétroaction
> robuste, n'a pas été démontrée pour le fraisage de pièces flexibles,
> où le caractère interrompu de la coupe rend le signal impulsionnel et
> la dynamique régénérative large bande.

---

## 3. État de l'art et démarcation

### 3.1 Contrôle actif des vibrations de pièces flexibles en fraisage

Ligne de travaux essentiellement **par rétroaction** : retour de vitesse
direct [2], LQG piézoélectrique [3], commande modale active [4], mode
glissant [5]. La revue de référence [1] classe l'amortissement actif
comme rétroaction et traite les méthodes fondées sur la vitesse de broche
(sélection de vitesse, SSV) séparément — **aucune catégorie de
compensation anticipative périodique apprise**.

### 3.2 Compensation anticipative apprise / répétitive en usinage

Émergente, mais **pré-planifiée** : feedforward appris assistant un
amortisseur inertiel [6], absorbeur de vibration virtuel accordé à la
fréquence de passage de dent supposée connue [7], compensation indexée
par l'angle rejouée depuis un **codeur** dans la littérature brevet [8].
**Aucun** n'analyse la robustesse à l'erreur de synchronisation.

### 3.3 Sensibilité à l'erreur de période (fondement du verrou)

La commande répétitive standard ne tolère qu'un écart de période
d'environ **±0,1 %** avant effondrement du rejet [9] ; la commande
répétitive en domaine angulaire corrige cela **avec un codeur** [10]. La
vitesse de broche fluctue de façon mesurable sous coupe interrompue [11]
(glissement d'une broche asynchrone ≈ 2–3 % à charge nominale) ; la SSV
volontaire atteint **5–30 %** [12] — un à deux ordres de grandeur au-delà
de la tolérance d'une compensation périodique synchronisée par horloge.

### 3.4 Estimation sans capteur de la phase/vitesse

L'estimation *tacholess* de vitesse à partir des vibrations est mature en
surveillance [13] ; la compensation synchrone référencée par la vibration
existe pour le **balourd mono-harmonique** en paliers magnétiques actifs
(AMB) [14]. Mais en usinage, « sans capteur » a jusqu'ici alimenté
l'**adaptation de la consigne de vitesse** [15], non la synchronisation
d'une compensation anticipative côté actionneur.

### 3.5 Tableau de démarcation

| Travail | Anticipatif appris | Milieu | Synchronisation | Robuste vitesse | Repli sécurisé |
|---|---|---|---|---|---|
| Parus 2013 [3], Wan 2020 [5] | non (rétroaction) | fraisage pièce flexible | — | — | — |
| Bahtiyar 2024 [6] | **oui** | usinage | horloge/déclencheur | non | non |
| Brevet Fanuc [8] | oui (angle) | broche | **codeur** | par codeur | non |
| RC angulaire [10] | répétitif | rotatif général | **codeur** | par codeur | non |
| AMB APF/notch [14] | mono-harmonique | rotor | **sans capteur** | oui | non |
| **PLAD (cette thèse)** | **oui** | **fraisage paroi mince** | **sans capteur (PLL)** | **oui** | **oui (gating)** |

La dernière ligne est le seul cas réunissant toutes les colonnes ; c'est
la démarcation revendiquée.

---

## 4. Contribution proposée : PLAD

```
u(t) = u_LQG(x̂) + α · c_lock(t) · NN_FF(φ̂(t), x̂)
```

1. **Observateur de phase de broche sans capteur.** Filtre passe-bande
   (Q = 4) centré sur la fréquence de passage de dent nominale, suivi
   d'une **PLL numérique** (détecteur produit + boucle PI) qui verrouille
   le fondamental forcé dans le signal de déplacement lui-même — aucun
   codeur. Plage de capture ±7 %, temps de verrouillage mesuré ≈ 0,15 s.
2. **Référencement de phase basé modèle, ordonnancé en position.** La FRF
   en boucle fermée entre le fondamental de l'effort de coupe et le
   capteur convertit la phase du *déplacement* en phase de l'*horloge de
   perturbation*, interpolée sur une grille (fréquence × position
   d'outil) — première exploitation du point d'entrée `enable_gs` du
   solveur. Une calibration unique hors ligne absorbe le biais résiduel.
3. **Gating de confiance — l'adaptation enfin connectée.** La qualité de
   verrouillage (direction du vecteur (I,Q) démodulé = cos Δθ,
   indépendante de l'amplitude) module continûment le gain anticipatif :
   pleine autorité verrouillé, dégradation douce vers le **LQG pur** en
   perte de verrouillage. La saturation de la butée de fréquence est
   détectée comme *pseudo-verrouillage*, de sorte que les déviations
   au-delà de ±7 % rétractent la compensation de façon déterministe. Ceci
   remplace le `lambda_robust` inerte du v3 par un signal d'adaptation qui
   atteint réellement la loi de commande, et se compose avec le filtre de
   sécurité de Lyapunov conservé.
4. **Poids appris identiques.** Le v4 réutilise verbatim le réseau entraîné
   du v3, de sorte que tout écart de performance mesuré est imputable à la
   **seule** couche de synchronisation.

---

## 5. Résultats de validation (ce dépôt)

`05_main/main_gap_spindle_sync.py` ; contrôleurs conçus/entraînés à la
vitesse nominale uniquement ; fenêtre établie t > 0,15 s ; résultats
complets dans `results_gap_sync/`. Réduction de RMS vs. la ligne de base
LQG du même scénario :

| Scénario | DARC v3 (horloge ouverte) | DARC v4 (PLAD) |
|---|---:|---:|
| Nominal (a_p = 0,3 mm) | +4,6 % | +4,7 % |
| Décalage +1,23 % | −0,1 % | +4,7 % |
| Décalage +2,50 % | +0,8 % | +5,6 % |
| Décalage −1,20 % | −0,2 % | +4,8 % |
| Fluctuation SSV ±1 % @ 2 Hz | −0,3 % | +5,1 % |
| Passe longue 4 s, +2,5 % | +0,8 % | +6,8 % |
| +2,5 % avec bruit capteur 0,1 µm | +1,0 % | +5,6 % |
| +9,3 % (hors plage de capture) | +1,7 % | −0,1 % (repli sur LQG) |

Une erreur de vitesse de 1–2,5 % **annule le bénéfice anticipatif** du v3
(moyenne des six scénarios hors-nominal : **+0,26 %**), tandis que le v4
le **conserve** (**+5,28 %** en moyenne), avec une confiance de
verrouillage de 0,98–1,00 et aucun coût en régime établi au nominal. Le
v3 désynchronisé **gaspille en outre de l'effort de commande** (ex.
scénario a_p = 0,6 mm : u_RMS 7,83 V pour v3 contre 7,34 V pour v4) tout
en vibrant davantage. La suite de validation
`03_analysis/validate_phase_observer.py` (12 vérifications) passe
intégralement.

---

## 6. Portée et limites

- **A1/A5 sont la condition d'entraînement** (train-on-test) : elles
  servent de meilleur cas pour v3, non de test de généralisation. L'objet
  de l'expérience est précisément l'écart entre entraînement et
  déploiement.
- La fenêtre établie exclut le transitoire mécanique *et* le verrouillage
  du v4 ; les RMS plein-enregistrement sont fournis dans `metrics.json`.
  « Aucun coût au nominal » se rapporte au régime verrouillé.
- L'observateur exige un fondamental forcé détectable près du nominal
  (plage de capture ±7 %). Au-delà (SSV agressive), la saturation de
  butée est détectée et le système se replie sur le LQG (scénario E) ;
  retenir la compensation demanderait un recentrage depuis la *consigne*
  de broche (toujours sans codeur).
- Preuve en simulation, capteur idéal dans les scénarios de base ; le
  scénario D exerce l'observateur sous bruit (0,1 µm RMS) + retard
  (50 µs).

---

## Références

[1] Munoa et al., *CIRP Annals* 65(2):785–808, 2016. doi:10.1016/j.cirp.2016.06.004
[2] Zhang & Sims, *Smart Mater. Struct.* 14(6):N65, 2005. doi:10.1088/0964-1726/14/6/N01
[3] Parus et al., *J. Vib. Control* 19(7):1103–1120, 2013. doi:10.1177/1077546312442097
[4] Du & Long, *J. Manuf. Processes*, 2022. S1526612522007551
[5] Wan et al., *Mech. Syst. Signal Process.* 136:106528, 2020. doi:10.1016/j.ymssp.2019.106528
[6] Bahtiyar, Sencer & Beudaert, *CIRP Annals* 73(1), 2024. S0007850624000210
[7] Franco et al., *CIRP Annals* 72(1), 2023. S0007850623000471
[8] Brevet US 9,846,428 (Fanuc), « Controller for spindle motor ».
[9] Steinbuch, *Automatica* 38(12):2103–2109, 2002. doi:10.1016/S0005-1098(02)00134-6
[10] *Automatica* 158:111282, 2023. doi:10.1016/j.automatica.2023.111282
[11] Soshi, Raymond & Ishii, *Procedia CIRP* 14:159–163, 2014. doi:10.1016/j.procir.2014.03.087
[12] Seguy et al., *Int. J. Adv. Manuf. Technol.* 50:883–895, 2010. doi:10.1007/s00170-009-2336-9
[13] Peeters et al., *Mech. Syst. Signal Process.* 129:407–436, 2019. doi:10.1016/j.ymssp.2019.02.031
[14] Xu, Wu & Guan, *Shock and Vibration* 2020:2606178, 2020. doi:10.1155/2020/2606178
[15] Yamato et al., *Int. J. Precis. Eng. Manuf.* 22:1071, 2021. doi:10.1007/s12541-021-00469-2
