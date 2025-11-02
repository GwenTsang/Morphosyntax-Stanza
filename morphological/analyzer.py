from typing import List, Dict

try:  # pragma: no cover - dépendance optionnelle
    import stanza
except ImportError:  # pragma: no cover - Stanza doit être installé
    stanza = None

try:
    import torch
except ImportError:  # pragma: no cover - torch peut être absent en CPU-only
    torch = None

from morphological.dictionaries import MorphologicalDictionary


class MorphologicalAnalyzer:
    def __init__(self, language: str = 'fr'):
        """
        Analyseur morpho-syntaxique basé sur Stanza.
        """
        self.language = language

        if stanza is None:
            raise ImportError(
                "Stanza n'est pas installé, installez-le."
            )

        # Téléchargement silencieux des ressources nécessaires (tokenize, mwt, pos)
        stanza.download(language, processors='tokenize,mwt,pos', verbose=False)

        use_gpu = bool(torch and torch.cuda.is_available())
        self.nlp = stanza.Pipeline(
            language,
            processors='tokenize,mwt,pos',
            use_gpu=use_gpu,
            tokenize_no_ssplit=True,
        )

        self.dictionary = MorphologicalDictionary()

    def analyze(self, text: str) -> List[Dict]:
        """Analyse morpho-syntaxique d'un texte (une seule chaîne)."""
        batch_result = self.analyze_batch([text])
        return batch_result[0] if batch_result else []

    def analyze_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Analyse morpho-syntaxique d'une liste de textes."""
        if not texts:
            return []

        print(f"[Analyzer:stanza] Processing batch of {len(texts)} texts")

        docs = [self.nlp(text) for text in texts]
        return [self._format_stanza_doc(doc) for doc in docs]

    def _analyze_stanza(self, text: str) -> List[Dict]:
        """Analyse avec Stanza (méthode interne)."""
        doc = self.nlp(text)
        return self._format_stanza_doc(doc)

    def _format_stanza_doc(self, doc) -> List[Dict]:
        tokens: List[Dict] = []

        for sent in doc.sentences:
            for word in sent.words:
                morph_features = {
                    'text': word.text,
                    'lemma': word.lemma,
                    'pos': word.upos,
                    'xpos': word.xpos,
                    'dep': word.deprel,
                    'head': word.head,
                    'features': {},
                }

                # Extraction des traits morphologiques (ex: Gender=Fem|Number=Sing)
                if word.feats:
                    for feat in word.feats.split('|'):
                        if '=' in feat:
                            key, value = feat.split('=', 1)
                            morph_features['features'][key.lower()] = value.lower()

                tokens.append(morph_features)

        return tokens
