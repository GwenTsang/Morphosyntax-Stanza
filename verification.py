import sys
from constraints.agreement import AgreementChecker
from morphological.analyzer import MorphologicalAnalyzer

def check_sentence(sentence: str) -> bool:
    analyzer = MorphologicalAnalyzer(tool="stanza", language="fr")
    checker = AgreementChecker(
        post_adj_max=1,
        max_total_adjs=10**9,
        max_consecutive_same_adj=1,
        require_det=True,
        require_noun=True,
    )
    tokens = analyzer.analyze(sentence)
    return (
        checker.check_anti_repetition(tokens)
        and checker.check_gender_agreement(tokens)
        and checker.check_number_agreement(tokens)
        and checker.check_subject_verb_agreement(tokens)
    )

def main():
    if len(sys.argv) < 2:
        print('Usage: python verification.py "Votre phrase en français."')
        sys.exit(1)
    sentence = " ".join(sys.argv[1:]).strip()
    ok = check_sentence(sentence)
    print("✓ Correct — " + sentence if ok else "✖ Incorrect — " + sentence)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
