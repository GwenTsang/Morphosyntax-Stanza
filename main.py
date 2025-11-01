import argparse

from morphological.analyzer import MorphologicalAnalyzer
from morphological.templates import MorphosyntacticTemplate

def main():
    parser = argparse.ArgumentParser(description='Morphosyntaxe sous contrainte')
    parser.add_argument('--mode', choices=['generate', 'decode', 'evaluate'], 
                       default='generate')
    parser.add_argument('--tool', choices=['spacy', 'stanza'], default='spacy')
    parser.add_argument('--dict-path', help='Chemin vers les dictionnaires')
    args = parser.parse_args()

    print("Analyzer Initialisation")
    analyzer = MorphologicalAnalyzer(tool=args.tool)
    print("Ready")
    
    if args.dict_path:
        analyzer.dictionary.load_morphalou(f"{args.dict_path}/morphalou.xml")
        analyzer.dictionary.load_lefff(f"{args.dict_path}/lefff.txt")
    
    if args.mode == 'generate':
        # Génération avec gabarits
        template_gen = MorphosyntacticTemplate(analyzer)
        
        # Exemple de génération
        lexical_items = {
            'DET': ['Le', 'La', 'Les'],
            'ADJ': ['petit', 'petite', 'grand', 'grande'],
            'NOUN': ['chat', 'chatte', 'chien', 'chienne']
        }
        
        sentences = template_gen.generate_with_constraints('SN', lexical_items)
        
        print("Phrases générées avec contraintes d'accord:")
        for sent in sentences[:10]:
            print(f"  - {sent}")
    
    elif args.mode == 'decode':
        # Décodage contraint
        from constraints.decoder import ConstrainedDecoder
        import numpy as np

        decoder = ConstrainedDecoder(analyzer)
        
        # Exemple de décodage
        vocabulary = ['Le', 'La', 'chat', 'mange', 'souris', 'petite']
        scores = np.random.random(len(vocabulary))
        
        results = decoder.beam_search_with_constraints(
            scores, vocabulary, beam_width=3, max_length=5
        )
        
        print("Résultats du décodage contraint:")
        for sequence, score in results:
            print(f"  - {' '.join(sequence)} (score: {score:.3f})")
    
    elif args.mode == 'evaluate':
        # Évaluation
        from constraints.agreement import AgreementChecker
        from utils.evaluation import GrammaticalityEvaluator

        checker = AgreementChecker()
        evaluator = GrammaticalityEvaluator(analyzer, checker)
        
        # Phrases de test
        test_sentences = [
            "Le petit chat noir",
            "La petite chat noir",  # Erreur d'accord
            "Les chats mangent",
            "Les chat mange"  # Erreur d'accord
        ]
        
        results = evaluator.evaluate_grammaticality(test_sentences)
        
        print(f"Résultats d'évaluation:")
        print(f"  - Taux de grammaticalité: {results['grammaticality_rate']:.2%}")
        print(f"  - Erreurs détectées: {dict(results['errors'])}")

if __name__ == "__main__":
    main()
