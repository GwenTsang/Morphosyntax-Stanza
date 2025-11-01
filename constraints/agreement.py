from typing import List, Dict, Optional

class AgreementChecker:
    """
    Vérifie : gabarit DET (ADJ){<=pre} NOUN (ADJ){<=post}, anti-répétition d'ADJ,
    accords (genre, nombre, sujet–verbe). Les tokens suivent le schéma :
    {"text":..., "lemma":..., "pos":..., "dep":..., "head":int, "features":{"gender":..,"number":..,"person":..}}
    """
    _POS_GENDER = {"DET", "ADJ", "NOUN"}
    _POS_NUMBER = {"DET", "ADJ", "NOUN", "VERB"}
    _DEP_SUBJ = {"nsubj", "nsubj:pass"}

    def __init__(self, *, pre_adj_max=1, post_adj_max=2,
                 max_total_adjs=3, max_consecutive_same_adj=1,
                 max_same_adj_per_sentence=1,
                 require_det=True, require_noun=True, default_subj_person="3"):
        self.pre_adj_max = pre_adj_max
        self.post_adj_max = post_adj_max
        self.max_total_adjs = max_total_adjs
        self.max_consecutive_same_adj = max_consecutive_same_adj
        self.max_same_adj_per_sentence = max_same_adj_per_sentence
        self.require_det = require_det
        self.require_noun = require_noun
        self.default_subj_person = default_subj_person

    # ---- Accords
    def check_gender_agreement(self, tokens: List[Dict]) -> bool:
        g = None
        for t in tokens:
            if t.get("pos") in self._POS_GENDER:
                tg = (t.get("features") or {}).get("gender")
                if tg:
                    if g is None: g = tg
                    elif g != tg: return False
        return True

    def _subjects_by_head(self, tokens: List[Dict]):
        m = {}
        for i, t in enumerate(tokens):
            if t.get("dep") in self._DEP_SUBJ and isinstance(t.get("head"), int):
                m.setdefault(t["head"], []).append(i)
        return m

    def check_number_agreement(self, tokens: List[Dict]) -> bool:
        subj_map = self._subjects_by_head(tokens)
        base = None
        for i, t in enumerate(tokens):
            if t.get("pos") not in self._POS_NUMBER: continue
            n = (t.get("features") or {}).get("number")
            if not n: continue
            if t.get("pos") == "VERB":
                for si in subj_map.get(i, []):
                    sn = (tokens[si].get("features") or {}).get("number")
                    if sn and sn != n: return False
            else:
                if base is None: base = n
                elif base != n: return False
        return True

    def check_subject_verb_agreement(self, tokens: List[Dict]) -> bool:
        subj_map = self._subjects_by_head(tokens)
        for vi, v in enumerate(tokens):
            if v.get("pos") != "VERB": continue
            vf = v.get("features") or {}
            for si in subj_map.get(vi, []):
                sf = (tokens[si].get("features") or {})
                if sf.get("number") and vf.get("number") and sf["number"] != vf["number"]:
                    return False
                sp = sf.get("person", self.default_subj_person)
                if vf.get("person") and sp != vf["person"]:
                    return False
        return True

    # ---- Gabarit syntaxique
    def check_syntax_template(self, tokens: List[Dict]) -> bool:
        if not tokens: return False
        pos = [t.get("pos") for t in tokens]
        i = 0
        if self.require_det:
            if i >= len(pos) or pos[i] != "DET": return False
            i += 1
        else:
            if i < len(pos) and pos[i] == "DET": i += 1
        pre = 0
        while i < len(pos) and pos[i] == "ADJ" and pre < self.pre_adj_max:
            i += 1; pre += 1
        if i < len(pos) and pos[i] == "ADJ" and pre >= self.pre_adj_max: return False
        if self.require_noun:
            if i >= len(pos) or pos[i] != "NOUN": return False
            i += 1
        else:
            if i < len(pos) and pos[i] == "NOUN": i += 1
        post = 0
        while i < len(pos) and pos[i] == "ADJ" and post < self.post_adj_max:
            i += 1; post += 1
        if i < len(pos): return False
        return True

    # Anti-répétition
    def check_anti_repetition(self, tokens: List[Dict]) -> bool:
        total_adjs = sum(1 for t in tokens if t.get("pos") == "ADJ")
        if total_adjs > self.max_total_adjs: return False
        run_lemma, run_len = None, 0
        lemma_counts = {}
        for t in tokens:
            if t.get("pos") != "ADJ":
                run_lemma, run_len = None, 0
                continue
            lemma = (t.get("lemma") or t.get("text") or "").strip().lower()
            if lemma and lemma == run_lemma:
                run_len += 1
            else:
                run_lemma, run_len = lemma, 1
            if run_len > self.max_consecutive_same_adj:  # p.ex. "petit petit petit" refuse, "petit petit" ok
                return False
            if lemma:
                lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1
                if lemma_counts[lemma] > self.max_same_adj_per_sentence:
                    return False
        return True

    # ---- Orchestrateur
    def validate(self, tokens: List[Dict]) -> bool:
        return (self.check_syntax_template(tokens)
                and self.check_anti_repetition(tokens)
                and self.check_gender_agreement(tokens)
                and self.check_number_agreement(tokens)
                and self.check_subject_verb_agreement(tokens))
