from typing import List, Dict

try:  # pragma: no cover - optional dependency
    import stanza
except ImportError:  # pragma: no cover - Stanza is optional when not selected
    stanza = None

try:
    import torch
except ImportError:  # pragma: no cover - torch may be absent in CPU-only setups
    torch = None

from morphological.dictionaries import MorphologicalDictionary


class MorphologicalAnalyzer:
    def __init__(self, tool='spacy', language='fr'):
        self.tool = tool
        self.language = language

        if tool == 'spacy':
            if spacy is None:
                raise ImportError(
                    "SpaCy is not installed. Install it or select another analyzer "
                    "with --tool."
                )

            self.nlp = spacy.load(f'{language}_core_news_lg')
        elif tool == 'stanza':
            if stanza is None:
                raise ImportError(
                    "Stanza is not installed. Install it or select another analyzer "
                    "with --tool."
                )

            stanza.download(language, processors='tokenize,mwt,pos', verbose=False)
            use_gpu = bool(torch and torch.cuda.is_available())
            self.nlp = stanza.Pipeline(
                language,
                processors='tokenize,mwt,pos',
                use_gpu=use_gpu,
                tokenize_no_ssplit=True,
            )
        else:
            raise ValueError(f"Unsupported tool: {tool}")

        self.dictionary = MorphologicalDictionary()

    def analyze(self, text: str) -> List[Dict]:
        """Analyse morphosyntaxique du texte"""
        batch_result = self.analyze_batch([text])
        return batch_result[0] if batch_result else []

    def analyze_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Analyse morphosyntaxique d'une liste de textes"""
        if not texts:
            return []

        print(f"[Analyzer:{self.tool}] Processing batch of {len(texts)} texts")

        if self.tool == 'spacy':
            docs = list(self.nlp.pipe(texts))
            return [self._format_spacy_doc(doc) for doc in docs]
        elif self.tool == 'stanza':
            docs = [self.nlp(text) for text in texts]
            return [self._format_stanza_doc(doc) for doc in docs]

        return []

    def _analyze_spacy(self, text: str) -> List[Dict]:
        """Analyse avec SpaCy"""
        doc = self.nlp(text)
        return self._format_spacy_doc(doc)

    def _format_spacy_doc(self, doc) -> List[Dict]:
        tokens = []

        for token in doc:
            morph_features = {
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'head': token.head.i,
                'features': {},
            }

            # Extraction des traits morphologiques
            for feat in token.morph:
                key, value = str(feat).split('=')
                morph_features['features'][key.lower()] = value.lower()

            tokens.append(morph_features)

        return tokens

    def _analyze_stanza(self, text: str) -> List[Dict]:
        """Analyse avec Stanza"""
        doc = self.nlp(text)
        return self._format_stanza_doc(doc)

    def _format_stanza_doc(self, doc) -> List[Dict]:
        tokens = []

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

                if word.feats:
                    for feat in word.feats.split('|'):
                        if '=' in feat:
                            key, value = feat.split('=')
                            morph_features['features'][key.lower()] = value.lower()

                tokens.append(morph_features)

        return tokens
