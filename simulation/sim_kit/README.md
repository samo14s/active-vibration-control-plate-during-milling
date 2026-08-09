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

## Reperes a ne pas depasser sans explication

| grandeur | valeur sans controle |
|---|---|
| limite de stabilite a 4900 tr/min | 0.0605 mm |
| RMS a 0.05 mm, 4900 tr/min | 0.2218 um |
| cout multi-vitesses a 0.25 mm | 11.137 |
| receptance au pic 536 Hz | 482.4 um/N |
| frequence de broutement simulee | 532 Hz (mesuree 580 Hz) |

Un cout de 12.000 signifie instable aux cinq vitesses. Un correcteur utile doit
descendre nettement sous 11.14 ; les valeurs de l'ordre de 1 sont atteignables.

Un correcteur qui affiche mieux que ces reperes **sans avoir rien change au
modele** doit etre suspecte avant d'etre publie.

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
