# GPT-2 Fine-tuning for Bruno Mars Style Lyrics Generation

This project fine-tunes the GPT-2 language model to generate song lyrics in the style of Bruno Mars.

## Project Structure

1. **Data Preparation**: 
   - Loads Bruno Mars lyrics from a text file
   - Preprocesses the text by ensuring proper spacing between songs
   - Saves processed data for training

2. **Model Setup**:
   - Uses the GPT-2 tokenizer with special handling for padding
   - Loads the base GPT-2 model
   - Configures the tokenizer to handle song lyrics

3. **Training Configuration**:
   - Sets appropriate hyperparameters for creative text generation
   - Configures training arguments with optimal settings for lyrics generation
   - Sets up data collator for language modeling

4. **Training**:
   - Fine-tunes GPT-2 on the Bruno Mars lyrics dataset
   - Saves checkpoints during training
   - Tracks training progress

5. **Generation**:
   - Loads the fine-tuned model
   - Generates lyrics based on various prompts
   - Uses sampling techniques for creative output

## Key Components Explained

### 1. Data Preparation
- The Bruno Mars lyrics are loaded from a text file
- We ensure each song is properly separated with double newlines
- This helps the model learn song structure and boundaries

### 2. Tokenization
- Uses GPT-2's tokenizer with special handling:
  - Sets the padding token to be the same as the end-of-sequence token
  - Limits sequence length to 128 tokens (shorter than standard 512 for lyrics)
  - Returns PyTorch tensors for efficient GPU processing

### 3. Model Configuration
- The base GPT-2 model is loaded with its pre-trained weights
- The token embeddings are resized to match our tokenizer
- This ensures the model can handle any special tokens we've added

### 4. Training Parameters
- **Learning Rate**: 5e-5 (slightly higher than standard for creative tasks)
- **Epochs**: 5 (more than typical for better style capture)
- **Batch Size**: 4 (adjust based on your GPU memory)
- **Warmup Steps**: 500 (helps stabilize early training)
- **Logging**: Every 500 steps to monitor progress

### 5. Generation Parameters
- **Temperature**: 0.9 (for more creative, less predictable output)
- **Top-k**: 50 (samples from the top 50 likely next words)
- **Top-p**: 0.95 (nucleus sampling for diverse but coherent output)
- **Max Length**: 150 tokens (for complete lyrical phrases)
- 
## Dependencies

- Transformers library from Hugging Face
- PyTorch
- Datasets library
- Pandas (for data handling)
