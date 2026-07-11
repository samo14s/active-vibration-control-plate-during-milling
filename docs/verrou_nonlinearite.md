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
**thermiques, harmoniques ou aléatoires** — et non sous l'excitation de
coupe **régénérative à retard** du fraisage (à la seule exception, très
récente, de [6], discutée ci-dessous).

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

`05_main/main_geometric_nonlinear.py` — paroi h = 2 mm, AL6061, 5600 tr/min.
Toutes les valeurs proviennent de `results_geom_nl/metrics.json`.

**A. Diagramme de bifurcation** (`fig01_bifurcation.png`). En dessous de
a_p = 0,12 mm : régime forcé **stable**, modèle linéaire ≈ von Kármán
(2,8 → 5,3 µm) — **le modèle linéaire est adéquat**. À la limite de
stabilité (a_p = 0,15 mm) le modèle **linéaire diverge** (amplitude coupée
à 40 mm) ; von Kármán **borne** la croissance en un **cycle limite**
(bifurcation de Hopf). Les cycles limites **stabilisés et dans le domaine de
validité** (w/h ≤ 0,4) valent 511 µm (a_p = 0,18 mm) et 718 µm
(a_p = 0,20 mm). Au-delà (a_p ≥ 0,25 mm) le cycle atteint w/h = 5–8, hors
validité von Kármán, et n'est reporté qu'à titre indicatif (points
« creux » sur la figure).

**B. Fréquence dépendant de l'amplitude** (`fig03_backbone.png`). Le
raidissement dépend **fortement de la condition aux limites**, ce qui
constitue le **critère de configuration** — trois cas, du physiquement
cohérent à l'idéalisation :

| configuration | BC | ω_nl/ω_l @ A/h=1 |
|---|---|---|
| cantilever (3 bords libres) | cohérente | **1,001** (négligeable) |
| paroi 3-bords (base + 2 côtés encastrés) | **cohérente** | **1,108** (modéré) |
| paroi « slot » (flexion cantilever + côtés bridés en membrane) | idéalisation (borne sup.) | 1,270 |

La configuration **physiquement cohérente** (paroi 3-bords encastrés, BC
transverse ET membranaire identiques) donne un raidissement **modéré mais
réel** (1,108). La configuration « slot » (côtés bridés en membrane
seulement, flexion restant cantilever — réalisable par des glissières
latérales sans contact) **isole l'étirement membranaire** et sert de
**borne supérieure** rendant le cycle limite bien visible ; c'est celle du
diagramme de bifurcation. Le cantilever pur (borne inférieure) montre un
effet négligeable — d'où l'encadrement.

**C. Contrôle actif** (`fig04_active_control.png`), a_p = 0,20 mm, LQG conçu
sur le modèle linéaire, saturation réaliste ±150 V :

| | modèle LINÉAIRE (prédiction) | modèle von Kármán (réponse vraie) |
|---|---|---|
| boucle ouverte | divergence (→ 44 mm, coupé) | cycle limite borné 718 µm |
| LQG ±150 V | **divergence** (aucune cible finie) | **borné** 670 µm |

Points importants (honnêteté) : **c'est la non-linéarité géométrique — non
le régulateur — qui borne la réponse** ; le LQG (aveugle au retard
régénératif : `lqg_controller.py` construit A,B,C sans le terme de retard)
ne réduit le cycle que de **6,7 %** (718 → 670 µm) à 91 % du budget de
tension. La conclusion n'est donc **pas** « le LQG stabilise », mais : le
modèle **linéaire ne fournit AUCUNE amplitude post-critique finie** (il
prédit une divergence sous commande), de sorte que **la validation de la
commande et le dimensionnement de l'actionneur en régime post-critique
exigent le modèle non linéaire**.

---

## 5. Portée et limites (déclarées honnêtement)

- **Tension flexibilité ↔ non-linéarité.** Une paroi assez souple pour
  broutter à w ~ h a, si elle est en cantilever, un raidissement faible
  (cycle limite grand) ; une paroi assez bridée pour un raidissement fort
  est rigide et broutte peu. Le cycle limite « modéré et borné » du
  diagramme de bifurcation est obtenu avec la configuration idéalisée
  « slot » (borne supérieure). La configuration **physiquement cohérente**
  (3-bords encastrés) donne un effet modéré (1,108) mais réel : la
  **conclusion robuste est le critère configuration/amplitude**, pas une
  amplitude de cycle limite universelle.
- **Le régulateur LQG ignore le retard régénératif** ; la divergence sous
  commande sur le plant linéaire tient en partie à ce choix de conception.
  L'argument porte donc sur l'**absence d'amplitude post-critique finie**
  dans le modèle linéaire, non sur une supériorité de la commande.
- **Modèle réduit à 3 modes** ; inertie dans le plan négligée (condensation
  statique), hypothèse standard de ces ROM.
- **Validité von Kármán** : rotations modérées ; seuls les points **stables
  et w/h ≤ 2** sont exploités quantitativement (a_p = 0,18–0,20 mm,
  w/h ≤ 0,36, w/L ≈ 0,009) ; les points w/h = 5–8 sont marqués hors
  validité sur la figure.
- **Contribution incrémentale, non disruptive** : le modèle réduit repose
  sur un bilan harmonique à 1 terme (même ordre qu'une réduction de Galerkin
  monomodale) ; l'apport propre est le **tenseur cubique assemblé par EF**
  (vs Galerkin analytique), la **comparaison modèle linéaire ↔ non
  linéaire** pour l'évaluation de la commande, et le **critère de
  configuration**. Les énoncés de type « jamais traité » sont à comprendre
  relativement à la littérature recensée, non comme un absolu vérifiable.
- **AL6061 homogène** ; extension **FGM/composites** (spécialité du
  rapporteur) = prolongement naturel.
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
