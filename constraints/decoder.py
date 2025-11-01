import numpy as np
from typing import List, Dict, Tuple
try:
    import pywrapfst as fst
    WFST_AVAILABLE = True
except ImportError:  # pragma: no cover - dépendance optionnelle indisponible
    class _MissingFstModule:
        """Fallback minimal pour informer de l'absence de pywrapfst."""

        class Fst:  # type: ignore
            pass

        @staticmethod
        def _raise():
            raise RuntimeError(
                "pywrapfst n'est pas installé. Les fonctionnalités WFST sont "
                "désactivées. Consultez la documentation du projet pour "
                "l'installer via OpenFst et le fork beer-asr."
            )

        compose = determinize = minimize = shortestpath = _raise

        @staticmethod
        def Compiler():
            _MissingFstModule._raise()

    fst = _MissingFstModule()  # type: ignore
    WFST_AVAILABLE = False

from morphological.analyzer import MorphologicalAnalyzer
from constraints.agreement import AgreementChecker

class ConstrainedDecoder:
    """Décodeur avec contraintes phonémiques/syntaxiques"""
    
    def __init__(self, analyzer: MorphologicalAnalyzer):
        self.analyzer = analyzer
        self.agreement_checker = AgreementChecker()
    
    def beam_search_with_constraints(self, 
                                    initial_scores: np.ndarray,
                                    vocabulary: List[str],
                                    beam_width: int = 5,
                                    max_length: int = 20,
                                    constraints: List[str] = None) -> List[Tuple[List[str], float]]:
        """
        Beam search avec contraintes morphosyntaxiques
        
        Args:
            initial_scores: scores initiaux pour chaque mot du vocabulaire
            vocabulary: liste des mots possibles
            beam_width: largeur du beam
            max_length: longueur maximale de la séquence
            constraints: liste des contraintes à appliquer
        
        Returns:
            Liste des k meilleures séquences avec leurs scores
        """
        constraints = constraints or ['gender_agreement', 'number_agreement']
        
        # Initialisation
        beams = [([], 0.0)]  # (sequence, score)
        
        for step in range(max_length):
            candidates = []
            
            for sequence, score in beams:
                # Pour chaque mot du vocabulaire
                for idx, word in enumerate(vocabulary):
                    new_sequence = sequence + [word]
                    new_score = score + initial_scores[idx]
                    
                    # Vérifier les contraintes
                    if self._check_constraints(new_sequence, constraints):
                        candidates.append((new_sequence, new_score))
            
            # Garder les k meilleurs
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            if not beams:
                break
        
        return beams
    
    def _check_constraints(self, sequence: List[str], 
                          constraint_names: List[str]) -> bool:
        """Vérifie si une séquence respecte les contraintes"""
        # Analyser la séquence
        text = ' '.join(sequence)
        tokens = self.analyzer.analyze(text)
        
        # Appliquer chaque contrainte
        for constraint in constraint_names:
            if constraint == 'gender_agreement':
                if not self.agreement_checker.check_gender_agreement(tokens):
                    return False
            elif constraint == 'number_agreement':
                if not self.agreement_checker.check_number_agreement(tokens):
                    return False
            elif constraint == 'subject_verb_agreement':
                if not self.agreement_checker.check_subject_verb_agreement(tokens):
                    return False
        
        return True
    
    def wfst_constrained_decode(self, input_fst: fst.Fst, 
                               constraint_fst: fst.Fst) -> fst.Fst:
        """
        Décodage avec WFST (Weighted Finite State Transducer)
        
        Args:
            input_fst: FST d'entrée
            constraint_fst: FST encodant les contraintes
        
        Returns:
            FST résultant après composition avec contraintes
        """
        if not WFST_AVAILABLE:
            raise RuntimeError(
                "Les fonctionnalités WFST nécessitent pywrapfst, qui n'est pas installé."
            )

        # Composition: applique les contraintes
        composed = fst.compose(input_fst, constraint_fst)
        
        # Optimisation
        composed = fst.determinize(composed)
        composed = fst.minimize(composed)
        
        # Trouver le meilleur chemin
        shortest_path = fst.shortestpath(composed)
        
        return shortest_path
    
    def build_constraint_fst(self, constraints: Dict) -> fst.Fst:
        """
        Construit un FST encodant les contraintes morphosyntaxiques
        
        Args:
            constraints: dictionnaire des contraintes
        
        Returns:
            FST des contraintes
        """
        if not WFST_AVAILABLE:
            raise RuntimeError(
                "Les fonctionnalités WFST nécessitent pywrapfst, qui n'est pas installé."
            )

        compiler = fst.Compiler()
        
        # État initial
        compiler.write('0 1 <eps> <eps> 0.0\n')
        
        state_id = 2
        
        # Encoder les contraintes d'accord
        if 'gender' in constraints:
            for gender in ['masc', 'fem']:
                compiler.write(f'1 {state_id} {gender} {gender} 0.0\n')
                compiler.write(f'{state_id} {state_id} {gender} {gender} 0.0\n')
                state_id += 1
        
        if 'number' in constraints:
            for number in ['sing', 'plur']:
                compiler.write(f'1 {state_id} {number} {number} 0.0\n')
                compiler.write(f'{state_id} {state_id} {number} {number} 0.0\n')
                state_id += 1
        
        # États finaux
        for i in range(1, state_id):
            compiler.write(f'{i}\n')
        
        return compiler.compile()
