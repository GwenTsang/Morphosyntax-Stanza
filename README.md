
Ce repo python ne fonctionne qu'avec `Stanza` et seulement en français.
Voir un exemple d'utilisation : https://colab.research.google.com/drive/14xd7QgkKvAJHSxzK4khHPK1khlIM5DuR?usp=sharing

La méthodologie fonctionne mais elle est un peu *sous-optimisée*, on pourrait faire un meilleur [élagage](https://fr.wikipedia.org/wiki/Élagage_alpha-bêta) dans la *beam search* .

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
## Tester la génération de phrases bien formées :
```python
!python main.py --mode generate
```

## Vérifier si une phrase est bien formée :


```python
python verification.py "Votre phrase"
```


## Bibliographie

Kahane, S., & Gerdes, K. (2023). Syntaxe théorique et formelle: Volume 1: Modélisation, unités, structures. Language Science Press.

https://en.wikipedia.org/wiki/Constraint_programming

https://perso.liris.cnrs.fr/christine.solnon/Site-PPC/e-miage-ppc-som.htm

