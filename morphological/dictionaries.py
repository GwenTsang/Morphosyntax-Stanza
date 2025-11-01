from typing import Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import xmltodict
except ImportError:  # pragma: no cover - xmltodict only required when loading Morphalou
    xmltodict = None

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - pandas only required when loading Lefff
    pd = None

class MorphologicalDictionary:
    def __init__(self):
        self.morphalou_data = {}
        self.lefff_data = {}
        
    def load_morphalou(self, path: str):
        """Charge le dictionnaire Morphalou depuis XML"""
        if xmltodict is None:
            raise ImportError(
                "xmltodict is required to load Morphalou resources. Install it to "
                "use load_morphalou()."
            )

        with open(path, 'r', encoding='utf-8') as f:
            data = xmltodict.parse(f.read())
            
        for entry in data['dictionary']['entry']:
            lemma = entry.get('lemma', '')
            forms = entry.get('inflected_form', [])
            if not isinstance(forms, list):
                forms = [forms]
                
            self.morphalou_data[lemma] = {
                'pos': entry.get('pos', ''),
                'forms': [{
                    'form': f.get('form', ''),
                    'gender': f.get('gender', ''),
                    'number': f.get('number', ''),
                    'tense': f.get('tense', ''),
                    'person': f.get('person', '')
                } for f in forms]
            }
    
    def load_lefff(self, path: str):
        """Charge le dictionnaire Lefff depuis fichier tabulé"""
        if pd is None:
            raise ImportError(
                "pandas is required to load Lefff resources. Install it to use "
                "load_lefff()."
            )

        df = pd.read_csv(path, sep='\t', names=['form', 'lemma', 'pos', 'features'])
        
        for _, row in df.iterrows():
            features = self._parse_features(row['features'])
            if row['lemma'] not in self.lefff_data:
                self.lefff_data[row['lemma']] = []
            
            self.lefff_data[row['lemma']].append({
                'form': row['form'],
                'pos': row['pos'],
                **features
            })
    
    def _parse_features(self, features_str: str) -> Dict:
        """Parse les traits morphologiques"""
        features = {}
        if pd is not None and pd.notna(features_str):
            for feat in features_str.split('|'):
                if '=' in feat:
                    key, value = feat.split('=')
                    features[key.lower()] = value.lower()
        return features
