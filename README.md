# PFE

### Mise en place / Setup :

Avoir installer FreeCad > 20.x
Installer les dependances python : 

```
pip install -r requirements.txt
```

Pré-requis dans FreeCad :
aller dans edit > preferences > general > macro et décocher 'run macro in local env...' : 

### Exécution :

#### Dans FreeCad

Charger 'pfe_gui.py' dans FreeCad

Executer avec Ctrl F6 pour créer la classe pfe

Executer des fonctions dans l'interface de script python...

Remarque : écrire 'pfe.' dans l'interface de script python permet de voir via un menu déroulant toutes les méthodes, et en completant de les appeler 

#### Standalone (sans ouvrir FreeCad)

`pfe_gui.py` utilise des fonctions de `pfe_standalone.py`.
Celle-cis n'ont pas besoin de la Gui de FreeCad et utilise FreeCad comme une librairie

`pfe_standalone.py` peut simplement etre importé, voir au début `pfe_gui.py` pour un exemple concret


### Fonctionnalitées implémentées

- Calcul des distances entre un object et un nuages de points

- Distance Map : plusieur classification disponible

- Feature Map : plusieurs versions plus ou moins rapide et/ou précise/exact

- Recalage de nuage de points avec CPD

- Bruit gaussien et generation de nuage

- Conversions vers format Open3D


