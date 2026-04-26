from language_model import LanguageModels, LanguageModelEvaluator

def main():
    manager = LanguageModels()
    if not manager.load():
        print("Training on full dataset...")
        manager.train() 
        manager.save()
    
    # Pass the manager directly, not just the models
    evaluator = LanguageModelEvaluator(manager)

    # Task 2: Next Word Prediction (Bigram only)
    print("\n" + "="*30 + " TASK 2 " + "="*30)
    context = "I have a house in"
    word, prob = evaluator.predict_next_word(context, 'bigram')
    print(f"Context: '{context}'")
    print(f"Predicted next word (Bigram): '{word}' (Log-Prob: {prob:.4f})")
    
    sentences = ["Brad Pitt was born in Oklahoma", "The actor was born in USA"]
    print(f"\n{'Task':<8} | {'Mode':<10} | {'Sentence':<30} | {'Log-Prob':<10} | {'Perp':<10}")
    print("-" * 85)

    for sent in sentences:
        lp3, pp3 = evaluator.evaluate_sentence(sent, 'bigram')
        print(f"{'Task 3':<8} | {'Bigram':<10} | {sent[:30]:<30} | {lp3:<10,.4f} | {pp3:<10,.4f}")
        
        lp4, pp4 = evaluator.evaluate_sentence(sent, 'smoothed')
        print(f"{'Task 4':<8} | {'Smoothed':<10} | {sent[:30]:<30} | {lp4:<10,.4f} | {pp4:<10,.4f}")
        print("-" * 85)

if __name__ == "__main__":
    main()