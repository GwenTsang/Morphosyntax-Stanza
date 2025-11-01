# Génération de phrases bien formées

Voir un exemple d'utilisation : https://colab.research.google.com/drive/14xd7QgkKvAJHSxzK4khHPK1khlIM5DuR?usp=sharing

La méthodologie fonctionne mais elle est *sous-optimisée*

## Etapes à faire manuellement :

```bash
pip install -r requirements.txt
```

```bash
!git clone https://github.com/beer-asr/pywrapfst.git
!cd pywrapfst
```

```python
!pip install -q stanza
import stanza
stanza.download('fr')
```


TO DO : Ajouter un `py` qui réutilise `agreements.py` pour vérifier si une phrase est bien formée au niveau des accords.

## Bibliographie

Kahane, S., & Gerdes, K. (2023). Syntaxe théorique et formelle: Volume 1: Modélisation, unités, structures. Language Science Press.

https://en.wikipedia.org/wiki/Constraint_programming

https://perso.liris.cnrs.fr/christine.solnon/Site-PPC/e-miage-ppc-som.htm

