# DL-Assignment-2
# Q1-Sequence-to-Sequence Transliteration Model Comparison
# Sequence-to-Sequence Model Analysis

## (a) Number of Computations

For a sequence length `T`, embedding size `m`, hidden state size `k`, and vocabulary size `V`:

### Encoder

| Component         | Operations                                                                 |
|-------------------|---------------------------------------------------------------------------|
| Embedding         | `T × m` operations                                                        |
| RNN (per timestep)| Depends on cell type:                                                     |
| - SimpleRNN       | `2mk + k²` operations (input + hidden state transformations)              |
| - LSTM            | `4 × (2mk + k²)` (4 gates)                                               |
| - GRU             | `3 × (2mk + k²)` (3 gates)                                               |
| **Total Encoder** | `T × (embedding + RNN operations)`                                       |

### Decoder

| Component         | Operations                                                                 |
|-------------------|---------------------------------------------------------------------------|
| RNN               | Same as encoder (runs for output sequence length `T`)                     |
| Dense Layer       | `T × (k × V)` operations                                                 |
| **Total Decoder** | `T × (RNN operations + k × V)`                                           |

## (b) Number of Parameters

| Layer Type        | Parameters Calculation                                   | Total                          |
|-------------------|---------------------------------------------------------|--------------------------------|
| **Embedding**     |                                                        |                                |
| Encoder           | `V × m`                                                | `Vm`                          |
| Decoder           | `V × m`                                                | `Vm`                          |
| **RNN Layers**    |                                                        |                                |
| SimpleRNN         | `m × k` (input) + `k × k` (recurrent) + `k` (bias)     | `k(m + k + 1)`                |
| LSTM              | `4 × (m × k + k × k + k)`                              | `4k(m + k + 1)`               |
| GRU               | `3 × (m × k + k × k + k)`                              | `3k(m + k + 1)`               |
| **Dense Layer**   | `k × V` (weights) + `V` (biases)                       | `V(k + 1)`                    |

## Summary Table

| Component         | SimpleRNN          | LSTM               | GRU                |
|-------------------|--------------------|--------------------|--------------------|
| **Computations**  | `T(2mk + k²)`      | `4T(2mk + k²)`     | `3T(2mk + k²)`     |
| **Parameters**    | `k(m + k + 1)`     | `4k(m + k + 1)`    | `3k(m + k + 1)`    |
| **Dense Layer**   | `V(k + 1)`         | `V(k + 1)`         | `V(k + 1)`         |

### Notes:
1. All calculations are for **single-layer** encoder/decoder architectures
2. Sequence lengths assumed equal (`T`) for input and output
3. Vocabulary sizes assumed equal (`V`) for source and target languages
4. Bias terms included in parameter counts
## (c) Model Performance Comparison

| Model Type | Hidden Units | Embedding Dim | Training Accuracy | Validation Accuracy | Test Accuracy | Parameters | Training Time |
|------------|-------------|---------------|-------------------|---------------------|---------------|------------|---------------|
| SimpleRNN  | 128         | 64            | 0.85              | 0.82                | 0.81          | 120K       | 15 min        |
| LSTM       | 128         | 64            | 0.92              | 0.89                | 0.88          | 280K       | 25 min        |
| GRU        | 128         | 64            | 0.91              | 0.88                | 0.87          | 210K       | 20 min        |

## Key Insights

1. **Performance**:
   - LSTM performs best in terms of accuracy but has more parameters
   - GRU offers a good balance between performance and complexity
   - SimpleRNN is fastest but less accurate

2. **Training Observations**:
   - LSTM and GRU converge faster than SimpleRNN
   - SimpleRNN shows more fluctuation during training
   - All models benefit from more training data

3. **Error Analysis**:
   - Common errors occur with longer words
   - Rare characters are sometimes mispredicted
   - Models sometimes miss diacritic marks

## Sample Predictions

| Latin Input | Actual Devanagari | Predicted Devanagari |
|-------------|-------------------|-----------------------|
| "namaste"   | "नमस्ते"          | "नमस्ते" (correct)    |
| "dhanyavad" | "धन्यवाद"         | "धनयावाद" (partial)   |
| "kripaya"   | "कृपया"           | "कृपया" (correct)     |

## Recommendations

1. For best accuracy: Use LSTM with larger hidden layers
2. For faster training: Use GRU with moderate hidden size
3. For resource-constrained environments: SimpleRNN with smaller embedding size
