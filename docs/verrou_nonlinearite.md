# Verrou scientifique — non-linéarité géométrique et contrôle actif

*Document rédigé pour insertion dans le manuscrit de thèse*
**« Contribution au contrôle actif des vibrations en fraisage des pièces
flexibles »**

Contribution mise en œuvre dans ce dépôt :
`01_core/von_karman_rom.py` (modèle réduit von Kármán cohérent EF),
`01_core/newmark_nonlinear_solver.py` (Newmark–Newton-Raphson),
démontrée par `05_main/main_geometric_nonlinear.py`, validée par
`03_analysis/validate_von_karman.py`.

---

## 1. Positionnement dans la thèse et vis-à-vis du rapporteur

Cette contribution se situe à l'intersection exacte du titre de la thèse et
de l'expertise reconnue du rapporteur **Pr. Mondher Wali** (ENIS, coques
intelligentes non linéaires, matériaux fonctionnellement gradués, éléments
finis géométriquement non linéaires — p.ex. Mallek, Jrad, Wali & Dammak,
*J. Intelligent Material Systems and Structures*, 2019, « Geometrically
nonlinear finite element simulation of smart laminated shells »).

| Élément | Réalisation |
|---|---|
| **Contrôle actif des vibrations** | patch piézoélectrique + LQG |
| **en fraisage** | effort de coupe régénératif à retard (fraise 3 dents) |
| **des pièces flexibles** | paroi mince AL6061 (homogène) |
| **non-linéarité géométrique** *(spécialité du rapporteur)* | plaque de von Kármán, couplage membrane-flexion, éléments finis |

La contribution mobilise **directement les outils du rapporteur** : la
cinématique de von Kármán, le couplage électro-mécanique du patch, la
réduction modale d'un modèle EF géométriquement non linéaire, la résolution
de Newton-Raphson.

---

## 2. Verrou scientifique

Deux littératures **matures mais disjointes** :

**(A) Contrôle actif du fraisage de parois minces → modèles LINÉAIRES.**
La quasi-totalité des travaux modélise la pièce par une plaque de
Kirchhoff/Mindlin réduite sur base modale LINÉAIRE : commande modale active
[1], PD variable dans le temps/espace [2], MPC [3], LQG/robuste [4],
perturbation robuste [5]. La non-linéarité géométrique de la paroi n'y
figure pas.

**(B) Contrôle actif GÉOMÉTRIQUEMENT NON LINÉAIRE (von Kármán) des
plaques/coques intelligentes FGM/piézolaminées** — littérature abondante et
mature [8–14] (la spécialité du rapporteur), mais sous chargements
**thermiques, harmoniques ou aléatoires**, JAMAIS sous l'excitation de
coupe **régénérative à retard** du fraisage.

> **Énoncé du verrou.** Le contrôle actif du broutement (chatter) en
> fraisage des parois minces est traité sur des modèles **linéaires**, qui
> ne peuvent décrire le comportement post-critique réel : au-delà de la
> limite de stabilité, le modèle linéaire prédit une **divergence
> exponentielle**, alors que la non-linéarité géométrique **borne** la
> croissance en un **cycle limite** (bifurcation de Hopf). La conséquence
> pour la commande — dimensionnement d'actionneur, validation de
> stabilisation, prédiction de l'amplitude résiduelle — n'a pas été établie
> sur un modèle **élément fini géométriquement non linéaire**, ni assortie
> d'un critère indiquant **quand** la non-linéarité doit être prise en
> compte.

**Démarcation honnête vis-à-vis de l'état de l'art le plus proche.**
Nasiri et al. (*MSSP*, 2025) [6] traitent la suppression du broutement
d'une plaque-pièce flexible **non linéaire** (déformations de von Kármán)
avec patches piézoélectriques et contrôleurs SAC/flou — c'est le travail le
plus proche. Il en diffère toutefois sur trois points, qui délimitent la
présente contribution :
1. le modèle non linéaire y est obtenu par **Galerkin/somme de modes
   analytique**, non par un **modèle EF cohérent** (approche du
   rapporteur) ;
2. il compare deux **lois de commande** sur le même modèle non linéaire ;
   il ne compare **pas** les prédictions du modèle **linéaire** et du
   modèle **non linéaire** pour une même commande — c.-à-d. l'erreur commise
   en utilisant le modèle linéaire pour dimensionner/valider la commande ;
3. il ne fournit **pas** de critère de configuration/amplitude délimitant le
   régime où la non-linéarité géométrique est significative.

> **Contribution nette.** Un **modèle réduit von Kármán cohérent EF** pour
> le contrôle actif du fraisage de parois minces, exploité pour (i)
> **quantifier quand** la non-linéarité géométrique compte (critère
> configuration/amplitude), et (ii) démontrer que le **modèle linéaire —
> norme du domaine — donne une évaluation QUALITATIVEMENT FAUSSE de la
> commande** (il prédit une divergence là où la réponse réelle est un cycle
> limite borné et contrôlable), conduisant à un **surdimensionnement** de
> l'actionneur.

---

## 3. Méthode (cohérente EF)

Modèle réduit non linéaire (coordonnées modales normées en masse) :

```
q̈_r + 2ζ_rω_r q̇_r + ω_r² q_r + Σ_ijk Γ_rijk q_i q_j q_k = F_ext,r
```

* la flexion transverse reste décrite par l'élément de Kirchhoff Q4
  existant (modes linéaires φ_r, pulsations ω_r) ;
* un élément **membranaire Q4 en contrainte plane** est ajouté sur le même
  maillage ; la déformation de von Kármán θ = {½w,x², ½w,y², w,x w,y}
  charge le problème membranaire, dont les ddl sont **condensés
  statiquement** (K_m a = −f_θ(w)) ;
* la force transverse non linéaire f_w = ∫ Gᵀ N dA (cubique en w) est
  projetée sur les modes → tenseur cubique **Γ** ;
* intégration temporelle **Newmark (β=¼, γ=½) + Newton-Raphson** interne, en
  conservant l'effort de coupe régénératif à retard et l'actionnement
  piézoélectrique.

**Validation** (`validate_von_karman.py`, 8/8) : force modale exactement
cubique (résidu 1e-15) ; jacobienne analytique vs différences finies
(1e-8) ; réduction linéaire exacte à petite amplitude ; et surtout **la
plaque carrée encastrée (immovable) reproduit la courbe maîtresse
classique ω_nl/ω_l ≈ 1,17 à A/h = 1** (littérature von Kármán, Chia,
Yamaki), ce qui atteste la cohérence du tenseur Γ.

---

## 4. Résultats (ce dépôt)

`05_main/main_geometric_nonlinear.py` — paroi h = 2 mm, AL6061, 5600 tr/min,
configuration « wall » (base encastrée + bords latéraux bridés en membrane,
représentant une nervure flanquée de matière épaisse).

**A. Diagramme de bifurcation** (`fig01_bifurcation.png`). En dessous de
a_p ≈ 0,12 mm : régime forcé stable, modèle linéaire ≈ von Kármán (quelques
µm) — **le modèle linéaire est adéquat**. Au-delà de la limite de stabilité
(a_p ≈ 0,15 mm) : le modèle **linéaire diverge** (croissance exponentielle
mesurée 12,5 → 129 → 14 000 µm), tandis que von Kármán **borne** la réponse
en un **cycle limite** (157 µm à l'amorce, croissant avec a_p).

**B. Fréquence dépendant de l'amplitude** (`fig03_backbone.png`). Le
raidissement dépend fortement de la condition membranaire : configuration
« wall » ω_nl/ω_l = 1,07 / 1,27 / 1,86 à A/h = 0,5 / 1 / 2 (fort), contre
cantilever pur (3 bords libres) ≈ 1,00 (négligeable) — d'où le **critère de
configuration**.

**C. Contrôle actif** (`fig04_active_control.png`), a_p = 0,20 mm, LQG conçu
sur le modèle linéaire, saturation réaliste ±150 V :

| | modèle LINÉAIRE (prédiction de conception) | modèle von Kármán (réponse vraie) |
|---|---|---|
| boucle ouverte | divergence (→ 44 mm, coupé) | cycle limite borné 718 µm |
| LQG ±150 V | **divergence** (non stabilisé) | **borné, stabilisé** 670 µm |

Un ingénieur s'appuyant sur le modèle **linéaire** conclurait que
l'actionneur ±150 V est **insuffisant** (réponse divergente) et
sur-dimensionnerait ; le modèle von Kármán révèle que la vibration
**s'auto-limite** en un cycle borné que la commande maintient — seul le
modèle géométriquement non linéaire permet le **dimensionnement correct**
et la **validation** de la commande active en régime post-critique.

---

## 5. Portée et limites

- **Configuration « wall »** : elle suppose des bords latéraux bridés en
  membrane (nervure entre matière épaisse). Le cantilever pur (3 bords
  libres) montre au contraire une non-linéarité négligeable dans la plage
  opérationnelle — les deux cas sont fournis, et **le critère
  configuration/amplitude est la conclusion d'ingénierie**.
- **Modèle réduit à 3 modes** : cohérent EF pour le couplage membranaire
  dominant ; l'inertie dans le plan est négligée (condensation statique),
  hypothèse standard pour ces ROM.
- **Validité von Kármán** : rotations modérées ; les cycles limites
  exploités restent à w/L ≲ 0,1 (point de fonctionnement A/h = 0,36,
  w/L ≈ 0,009). Aux a_p élevés le cycle limite dépasse cette plage et n'est
  utilisé que qualitativement.
- **AL6061 homogène** ; l'extension aux parois **FGM/composites** (spécialité
  du rapporteur) renforcerait le raidissement et constitue le prolongement
  naturel.
- Preuve en **simulation** ; capteur idéal.

---

## Références

[1] Du et al., « Chatter suppression for milling of thin-walled workpieces
based on active modal control », *J. Manuf. Processes*, 2022.
[2] Wang et al., « Vibration suppression of thin-walled workpiece milling
using a time-space varying PD control », *Int. J. Adv. Manuf. Technol.*,
2019.
[3] Li et al., « Model predictive control based active chatter control in
milling process », *MSSP*, 2019.
[4] Li et al., « Active control of milling chatter considering the coupling
effect of spindle-tool and workpiece systems », *MSSP*, 2021.
[5] Zhang et al., « Robust active control based milling chatter suppression
with perturbation model via piezoelectric stack actuators », *MSSP*, 2019.
[6] Nasiri et al., « Chatter suppression in nonlinear milling of a flexible
plate-workpiece with attached piezoelectric actuators : SAC vs optimized
type-2 fuzzy controller », *MSSP*, 2025. *(état de l'art le plus proche)*
[7] Mallek, Jrad, Wali & Dammak, « Geometrically nonlinear finite element
simulation of smart laminated shells… », *J. Intelligent Material Systems
and Structures*, 2019. *(travaux du rapporteur — cadre EF)*
[8] Fakhari et al., « Nonlinear vibration control of functionally graded
plate with piezoelectric layers in thermal environment », *J. Vibration and
Control*, 2011.
[9] Kattimani & Ray, « Control of geometrically nonlinear vibrations of
functionally graded magneto-electro-elastic plates », *Int. J. Mech. Sci.*,
2015.
[10] Mao & Fu, « Nonlinear dynamic response and active vibration control for
piezoelectric functionally graded plate », *J. Sound and Vibration*, 2010.
[11] Zhang et al., « Nonlinear thermo-electro-mechanical responses and
active control of functionally graded piezoelectric plates… »,
*Thin-Walled Structures*, 2024.
[12] Susheel et al., « Active shape and vibration control of functionally
graded thin plate using functionally graded piezoelectric material »,
*J. Intelligent Material Systems and Structures*, 2017.
[13] Jiang et al., « Nonlinear vibrations … piezoelectric FG graphene-
reinforced laminated composite cantilever plate with PPF control »,
*Thin-Walled Structures*, 2023.
[14] Chai et al., « Analysis and active control of nonlinear vibration of
composite lattice sandwich plates », *Nonlinear Dynamics*, 2020.
[15] Brand et al., « An active tool holder and robust LPV control design for
practical vibration suppression in internal turning », *Control Eng.
Practice*, 2025.
[16] Chia, *Nonlinear Analysis of Plates*, McGraw-Hill, 1980 (courbe
maîtresse von Kármán, référence de validation).
