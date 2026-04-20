Self-Pruning Neural Network — Report

Author: Thrinasoni R
Dataset: CIFAR-10
Framework: PyTorch

1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?
Setup

Each weight w in the network is paired with a learnable gate:

gate  =  sigmoid(gate_score)   ∈ (0, 1)
output =  (weight × gate) · input  +  bias

The training objective is:

Total Loss  =  CrossEntropyLoss  +  λ × SparsityLoss
SparsityLoss  =  Σ sigmoid(gate_score_i)
Why L1 Encourages Sparsity

The gradient of the sparsity loss is always positive, which pushes gate_score in the negative direction:

sigmoid(gate_score) → 0

This effectively removes the corresponding weight.

L1 vs L2
Regularizer	Behaviour
L1	        Continues pushing values to zero → promotes sparsity
L2	        Weakens near zero → values rarely become exactly zero

Thus, L1 is preferred for pruning because it can drive parameters to near-zero effectively.

Practical Note

In practice, sigmoid outputs rarely reach exact zero due to their smooth nature.
Therefore, sparsity is interpreted as near-zero gate values rather than exact zeros.

2. Results Table
Lambda	Test Accuracy (%)	Sparsity Level (%)
1e-5	46.89%	                        0.02%             
1e-4	47.74%	                        0.17%
1e-3	48.26%	                        3.33%
Observations
As λ increases, sparsity increases gradually.
Accuracy remains relatively stable, indicating that pruning is conservative.
The model removes only a small number of less important weights.

This shows that the network learns which connections are less useful without significantly harming performance.

3. Gate Value Distribution Plot




Interpretation
Most gate values remain away from zero → important weights retained
A small portion moves closer to zero → weak connections suppressed
This indicates controlled and gradual pruning behavior
4. Architecture
Input Image (32 × 32 × 3)
        ↓
Flatten (3072)
        ↓
PrunableLinear(3072 → 512) + ReLU
        ↓
PrunableLinear(512 → 256) + ReLU
        ↓
PrunableLinear(256 → 128) + ReLU
        ↓
PrunableLinear(128 → 10)
        ↓
Output (10 classes)

Each layer contains:

weights
bias
gate_scores (learnable pruning parameters)
5. How to Run
# Run in Google Colab (GPU recommended)
!python self_pruning_network.py

Outputs generated:

results_table.csv
gate_distribution.png

6. Key Takeaways
The model integrates pruning directly into training using differentiable gates.
L1-based regularization enables the model to suppress unnecessary connections.
λ controls the trade-off between accuracy and sparsity.
The model demonstrates gradual and stable pruning behavior rather than aggressive removal.
In practice, sparsity is defined using a threshold since exact zeros are rare in sigmoid outputs.

7. Conclusion

This approach demonstrates how neural networks can adaptively reduce their complexity during training.
Even with moderate sparsity levels, the model maintains performance, showing that many weights are redundant and can be pruned without significant accuracy loss.
