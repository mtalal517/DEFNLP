"""
Training Module for BERT QA Model
Fine-tunes the BERT model on the dataset if needed.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from typing import List, Dict, Tuple
import config
import utils


class QADataset(Dataset):
    """Dataset for Question Answering fine-tuning."""
    
    def __init__(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize QA dataset.
        
        Args:
            contexts: List of context texts
            questions: List of questions
            answers: List of answer dictionaries with 'text' and 'answer_start'
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Flatten tensors
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Find answer positions in tokenized text
        answer_text = answer['text']
        answer_start = answer['answer_start']
        
        # Encode answer separately to find token positions
        answer_encoding = self.tokenizer(
            answer_text,
            add_special_tokens=False
        )
        
        # Find start and end positions
        # This is a simplified approach; production code would need more robust handling
        start_positions = torch.tensor([1])  # Placeholder
        end_positions = torch.tensor([1])    # Placeholder
        
        encoding['start_positions'] = start_positions
        encoding['end_positions'] = end_positions
        
        return encoding


class BERTQATrainer:
    """Trainer for BERT QA model fine-tuning."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize trainer.
        
        Args:
            model_name: Name of pre-trained model to use
        """
        self.model_name = model_name or config.QA_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
        print(f"Loaded model: {self.model_name}")
    
    def prepare_training_data(self, train_df: pd.DataFrame) -> Tuple[List, List, List]:
        """
        Prepare training data from DataFrame.
        
        Args:
            train_df: Training DataFrame with text and labels
        
        Returns:
            Tuple of (contexts, questions, answers)
        """
        contexts = []
        questions = []
        answers = []
        
        # Load publication texts
        pub_texts = utils.load_json_publications(
            config.TRAIN_JSON_DIR,
            train_df['Id'].unique().tolist()
        )
        
        # Create training examples
        for idx, row in train_df.iterrows():
            pub_id = row['Id']
            dataset_title = row.get('dataset_title', '')
            
            if pub_id not in pub_texts or not dataset_title:
                continue
            
            context = pub_texts[pub_id]
            
            # Use multiple questions
            for question in config.QA_QUESTIONS:
                # Find answer in context
                answer_start = context.lower().find(dataset_title.lower())
                
                if answer_start != -1:
                    contexts.append(context)
                    questions.append(question)
                    answers.append({
                        'text': dataset_title,
                        'answer_start': answer_start
                    })
        
        print(f"Prepared {len(contexts)} training examples")
        return contexts, questions, answers
    
    def train(
        self,
        train_df: pd.DataFrame,
        output_dir: str = "./models/qa_model",
        num_epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None
    ):
        """
        Fine-tune the BERT QA model.
        
        Args:
            train_df: Training DataFrame
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Use config defaults if not provided
        num_epochs = num_epochs or config.QA_NUM_EPOCHS
        batch_size = batch_size or config.QA_BATCH_SIZE
        learning_rate = learning_rate or config.QA_LEARNING_RATE
        
        print("\n" + "="*60)
        print("FINE-TUNING BERT QA MODEL")
        print("="*60)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        # Prepare data
        contexts, questions, answers = self.prepare_training_data(train_df)
        
        # Create dataset
        dataset = QADataset(
            contexts=contexts,
            questions=questions,
            answers=answers,
            tokenizer=self.tokenizer,
            max_length=config.QA_MAX_SEQ_LENGTH
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=1000,
            save_total_limit=2,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save model
        print(f"\nSaving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("Training complete!")


def main():
    """Main entry point for training."""
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(config.TRAIN_CSV)
    
    # Create trainer
    trainer = BERTQATrainer()
    
    # Fine-tune model
    trainer.train(train_df)


if __name__ == "__main__":
    main()
