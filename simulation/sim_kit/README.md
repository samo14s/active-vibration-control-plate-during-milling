# Base de simulation — fraisage de paroi mince avec patch piezoelectrique

Modele fige et valide, **sans correcteur**. A utiliser pour comparer des lois de
commande sur un banc identique.

## Contenu

| fichier | role |
|---|---|
| `simulation_base.py` | interface unique : `SimBase`, `check_model` |
| `plate_model.py` | modele modal de la plaque, patch, capteur |
| `milling_force.py` | coefficients de coupe, efforts periodiques |
| `newmark_solver.py` | integration temporelle avec retard regeneratif |
| `fopid_controller.py` | operateurs fractionnaires (utile si votre correcteur en emploie) |
| `example_controller.py` | exemple minimal d'interface |
| `*.xlsx` | courbes experimentales numerisees ayant servi a la validation |

## Demarrage

```bash
python simulation_base.py      # verifie le modele et affiche ses proprietes
python example_controller.py   # montre comment brancher un correcteur
```

## Interface correcteur

```python
class MonCorrecteur:
    def __init__(self, dt, ...): ...
    def step(self, x_hat_prev, u_prev, y_meas):
        # y_meas en metres, retourne (x_hat, u) avec u en volts
        return x_hat_prev, u
    def reset(self): ...
```

Le simulateur sature deja a +/- 150 V. Si votre correcteur sature en interne,
utilisez la meme borne, sinon son observateur verra une commande differente de
celle reellement appliquee.

## Trois appels suffisent

```python
sim = SimBase()

r = sim.run(ctrl, rpm=4900, ap=0.25e-3, T=0.20)      # un essai
J = sim.multi_speed_cost(lambda dt, tau: MonCtrl(dt))  # cout de reglage
b = sim.stability_limit(lambda dt, tau: MonCtrl(dt))   # profondeur limite
f, G = sim.receptance(lambda dt, tau: MonCtrl(dt))     # reponse frequentielle
```

## Ce qui est fige, et pourquoi

* **dt = tau/82 exactement.** Le retard regeneratif doit tomber sur la grille
  d'echantillonnage. Ce n'est pas un reglage libre.
* **Cinq modes.** Justifie par les quatre antiresonances mesurees : un modele a
  trois modes n'en reproduit que deux.
* **Patch au coin inferieur gauche, capteur au coin oppose.** Deplacer l'un ou
  l'autre change le classement modal et invalide toute comparaison anterieure.
* **Convention de signe des efforts = Eq. (13) de l'article.** L'article se
  contredit entre son Eq. (13) et l'enchainement de ses Eqs. (1)(2)(5)(10)(A.4).
  Les deux conventions donnent des diagrammes de lobes quasiment
  complementaires, et seule l'Eq. (13) reproduit les Fig. 13(b), 14(a) et 18 de
  l'article. Voir `FORCE_SIGN` dans `simulation_base.py` ;
  `SimBase(force_sign=-1.0)` permet de le verifier.

## Reperes a ne pas depasser sans explication

| grandeur | valeur sans controle |
|---|---|
| limite de stabilite a 4900 tr/min | 0.0605 mm |
| RMS a 0.05 mm, 4900 tr/min | 0.2216 um |
| cout multi-vitesses a 0.25 mm | 11.144 |
| receptance au pic 536 Hz | 480.7 um/N |
| frequence de broutement simulee | 524.6 Hz (voir la note ci-dessous) |

Un cout de 12.000 signifie instable aux cinq vitesses. Un correcteur utile doit
descendre nettement sous 11.14 ; les valeurs de l'ordre de 1 sont atteignables.

Un correcteur qui affiche mieux que ces reperes **sans avoir rien change au
modele** doit etre suspecte avant d'etre publie.

### Sur la frequence de broutement : les deux premiers modes sont a egalite

Ne pas conclure d'un seul chiffre. Par la theorie moyennee d'ordre 0 sur ce
modele, mode 1 seul donne 0.0495 mm et mode 2 seul 0.0463 mm : 7 % d'ecart,
contre 0.94 a 1.61 mm pour les modes 3 a 5. Selon l'analyse menee, l'un ou
l'autre l'emporte :

| analyse | mode critique | frequence |
|---|---|---|
| lobes moyennes d'ordre 0 | 2 | 1070 Hz |
| simulation temporelle (4900 tr/min, ap = 0.30 mm) | 1 | 524.6 Hz |
| experience de Du, Fig. 20 | les deux | 580 et 1123 Hz |

Les 1135 Hz de la simulation publiee correspondent au mode que la theorie
lineaire de ce modele designe elle aussi : ce n'est pas une erreur de mode.

## Ce que les courbes experimentales ne valident PAS

Les deux courbes numerisees valident la **forme** de la reponse : cinq
resonances, quatre antiresonances, repartition des creux du transfert en
tension. Elles ne valident **pas son niveau**.

Le plateau basse frequence de la Fig. 12(a), lu avec la reference annoncee
(1 um/N), vaut 43.8 um/N ; une plaque de Kirchhoff aux dimensions du Tableau 1
donne 6.90 um/N par resolution statique EF exacte. Le facteur 6.36 ne vient pas
du modele : les memes matrices reproduisent les cinq frequences a 2 % pres, et
la souplesse est leur inverse — une raideur 6.36 fois trop faible donnerait
f1 = 214 Hz au lieu de 540 Hz. Les lobes de stabilite de l'article (Fig. 13)
s'accordent d'ailleurs avec la valeur raide. L'echelle en dB de la Fig. 12 est
donc inexploitable.

Consequence directe : **le niveau de `H_Pe` n'est pas calibre**, puisqu'il ne
pourrait l'etre que par le rapport des deux courbes. Les 480.7 um/N du tableau
ci-dessus sont une sortie de modele, pas une mesure. Toute performance en
boucle fermee herite de cette incertitude ; annoncez-la en balayant le gain :

```python
sim_nom = SimBase()                  # synthese du correcteur
sim_lo  = SimBase(gain_H=0.5)        # simulation, gain sous-estime
sim_hi  = SimBase(gain_H=2.0)        # simulation, gain surestime
```

`check_model()` verifie la coherence interne du niveau (statique modale contre
statique EF exacte, et contre la borne poutre) sans rien emprunter a la Fig. 12.
`run_demo.py --full` inclut les lignes `H x0.50` et `H x2.00`.

## Pieges rencontres, a ne pas refaire

1. **L'extraction lineaire de la fonction de sensibilite est invalide** pour les
   correcteurs a observateur : leur dynamique propre est souvent instable, leur
   reponse impulsionnelle diverge, et sa TFD n'a aucun sens. Utilisez
   `sim.receptance()`, qui mesure sur le simulateur non lineaire.
2. **Un reglage a une seule vitesse ne se transpose pas.** Un correcteur regle a
   4900 tr/min peut etre instable a cinq vitesses sur huit. Utilisez
   `multi_speed_cost`.
3. **L'horizon T est un parametre de cout, pas un detail.** Pres du seuil,
   passer de 0.30 s a 0.40 s deplace la limite mesuree d'environ 20 %.
4. **Classer les modes par |H_i D_i| est trompeur** : c'est un gain modal, pas
   une reponse. Divisez par omega_i^2. `sim.describe()` affiche les deux.

## Note sur la colocalisation

Le patch occupe le coin inferieur gauche, le capteur le coin superieur droit
oppose. La paire n'est donc PAS colocalisee : le retour de vitesse direct ne
beneficie d'aucune garantie de passivite, et `example_controller.py` montre
qu'il degrade effectivement la limite de stabilite. Tout correcteur efficace sur
cette configuration devra passer par une estimation d'etat.

## Proprietes modales, pour regler un observateur

`sim.describe()` affiche pour chaque mode : la frequence, l'entree patch H_i,
la sortie capteur D_obs,i, la participation a l'outil D_tool,i, puis les deux
classements |H_i D_i| et |H_i D_i|/omega_i^2.

Les deux classements different fortement, et c'est le second qui compte : le
produit brut est un gain modal, pas une reponse. Sur cette configuration le brut
place les modes 4 et 5 en tete alors qu'ils portent moins de 1 % de l'energie
d'excitation mesuree en coupe.
