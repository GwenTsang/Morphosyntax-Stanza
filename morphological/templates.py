from typing import Dict, List, Optional

from collections import defaultdict

from constraints.agreement import AgreementChecker
from morphological.analyzer import MorphologicalAnalyzer


class MorphosyntacticTemplate:
    def __init__(self, analyzer: MorphologicalAnalyzer):
        self.analyzer = analyzer
        self.agreement_checker = AgreementChecker()
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """Charge les gabarits prédéfinis"""
        return {
            'SN': {
                'pattern': 'DET? ADJ* NOUN ADJ*',
                'constraints': ['gender_agreement', 'number_agreement']
            },
            'SV': {
                'pattern': 'NOUN VERB',
                'constraints': ['subject_verb_agreement']
            },
            'SVO': {
                'pattern': 'NOUN VERB DET? NOUN',
                'constraints': ['subject_verb_agreement', 'object_agreement']
            }
        }

    def generate_with_constraints(
        self,
        template_name: str,
        lexical_items: Dict[str, List[str]],
        chunk_size: int = 512,
    ) -> List[str]:
        """Génère des phrases selon un gabarit avec contraintes d'accord."""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")

        generated: List[str] = []
        pattern_parts = template['pattern'].split()
        lexical_analyses = self._precompute_lexical_analyses(lexical_items)

        candidate_sequences = self._generate_candidate_sequences(
            pattern_parts,
            lexical_items,
            template['constraints'],
            lexical_analyses,
        )
        total_candidates = len(candidate_sequences)

        print(
            f"[Template:{template_name}] Generated {total_candidates} candidate sequences"
        )

        if not candidate_sequences:
            return generated

        total_chunks = (total_candidates + chunk_size - 1) // chunk_size

        for start in range(0, total_candidates, chunk_size):
            chunk = candidate_sequences[start:start + chunk_size]
            chunk_index = start // chunk_size + 1
            print(
                "[Template:{}] Analyzing chunk {}/{} (size={})".format(
                    template_name, chunk_index, total_chunks, len(chunk)
                )
            )
            texts = [' '.join(sequence) for sequence in chunk]
            analyses = self.analyzer.analyze_batch(texts)

            for sequence, tokens in zip(chunk, analyses):
                if not tokens or not sequence:
                    continue

                if self._check_constraints(tokens, template['constraints']):
                    generated.append(' '.join(token['text'] for token in tokens))

        return generated

    def _generate_candidate_sequences(
        self,
        pattern: List[str],
        lexical_items: Dict[str, List[str]],
        constraints: List[str],
        lexical_analyses: Dict[str, Dict[str, List[Dict]]],
        max_repetitions: int = 3,
    ) -> List[List[str]]:
        """Génère les séquences candidates sans analyse morphologique."""
        candidates: List[List[str]] = []

        def backtrack(
            current_words: List[str],
            current_tokens: List[Dict],
            current_pos_sequence: List[str],
            pos_index: int,
            repeat_count: int = 0,
            adj_counts: Optional[Dict[str, int]] = None,
        ) -> None:
            if adj_counts is None:
                adj_counts = {}
            if pos_index == len(pattern):
                if current_words:
                    candidates.append(current_words.copy())
                return

            token_spec = pattern[pos_index]
            pos = token_spec.rstrip('?*')
            is_optional = token_spec.endswith('?')
            is_multiple = token_spec.endswith('*')

            available_words = lexical_items.get(pos, [])

            if available_words:
                for word in available_words:
                    word_tokens = lexical_analyses.get(pos, {}).get(word)

                    if not word_tokens:
                        continue

                    if pos == 'ADJ':
                        current_count = adj_counts.get(word, 0)
                        if current_count >= 2:
                            continue
                        consecutive_identical = 0
                        idx = len(current_words) - 1

                        while (
                            idx >= 0
                            and current_pos_sequence[idx] == 'ADJ'
                            and current_words[idx] == word
                        ):
                            consecutive_identical += 1
                            idx -= 1

                        if consecutive_identical >= 2:
                            continue

                    current_words.append(word)
                    current_tokens.extend(word_tokens)
                    current_pos_sequence.append(pos)

                    updated_adj_counts = adj_counts
                    if pos == 'ADJ':
                        updated_adj_counts = adj_counts.copy()
                        updated_adj_counts[word] = current_count + 1

                    if not self._should_prune(current_tokens, constraints):
                        if is_multiple and repeat_count + 1 <= max_repetitions:
                            backtrack(
                                current_words,
                                current_tokens,
                                current_pos_sequence,
                                pos_index,
                                repeat_count + 1,
                                updated_adj_counts,
                            )

                        backtrack(
                            current_words,
                            current_tokens,
                            current_pos_sequence,
                            pos_index + 1,
                            0,
                            updated_adj_counts,
                        )

                    del current_tokens[-len(word_tokens):]
                    current_words.pop()
                    current_pos_sequence.pop()

            if is_optional or is_multiple:
                backtrack(
                    current_words,
                    current_tokens,
                    current_pos_sequence,
                    pos_index + 1,
                    0,
                )

        backtrack([], [], [], 0, 0, {})
        return candidates

    def _precompute_lexical_analyses(
        self, lexical_items: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """Pré-calcul des analyses morphologiques pour accélérer le pruning."""
        analyses_by_pos: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)

        for pos, words in lexical_items.items():
            if not words:
                continue

            # Conserver l'ordre mais éviter d'analyser plusieurs fois la même entrée
            unique_words = list(dict.fromkeys(words))
            analyses = self.analyzer.analyze_batch(unique_words)

            for word, tokens in zip(unique_words, analyses):
                analyses_by_pos[pos][word] = tokens

        return analyses_by_pos

    def _should_prune(self, tokens: List[Dict], constraints: List[str]) -> bool:
        """Vérifie si les contraintes incrémentales sont violées."""
        incremental_constraints = {
            'gender_agreement',
            'number_agreement',
        }

        for constraint in constraints:
            if constraint not in incremental_constraints:
                continue

            if constraint == 'gender_agreement':
                if not self.agreement_checker.check_gender_agreement(tokens):
                    return True
            elif constraint == 'number_agreement':
                if not self.agreement_checker.check_number_agreement(tokens):
                    return True

        return False

    def _check_constraints(self, tokens: List[Dict], constraints: List[str]) -> bool:
        """Applique les contraintes d'accord configurées pour le template."""
        for constraint in constraints:
            if constraint == 'gender_agreement':
                if not self.agreement_checker.check_gender_agreement(tokens):
                    return False
            elif constraint == 'number_agreement':
                if not self.agreement_checker.check_number_agreement(tokens):
                    return False
            elif constraint == 'subject_verb_agreement':
                if not self.agreement_checker.check_subject_verb_agreement(tokens):
                    return False
            else:
                # Contrainte inconnue : considérer comme satisfaite pour ne pas bloquer.
                continue

        if not self.agreement_checker.check_anti_repetition(tokens):
            return False

        return True
