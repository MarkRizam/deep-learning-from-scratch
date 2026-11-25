# AIR-004 — Neural networks and explicit backpropagation

## Abstract
We implemented educational float64 NumPy neural network layers and explicit gradients, and compared them with PyTorch and numerical derivatives. Eight synthetic experiments use seeds 11, 22 and 33; all 63 records reproduced within absolute tolerance 1e-9 in two local runs. Thirty local tests and eight notebooks passed. This is an educational reproduction, not a novel or peer-reviewed study.

## Methodology
See [protocol](protocol.md) for fixed configurations, datasets and baselines. Dense layers implement Y=XW+b; given upstream G, dW=X^T G, db=sum(G), dX=GW^T. Cross-entropy uses shifted logits; convolution is valid stride-one cross-correlation with explicit gradients. Training uses shuffled mini-batches, fixed epochs and independent initializations. PyTorch is not used in the learning implementation. All reported variation is sample SD across three seeds. No real-world datasets.

## Results
Mean maximum discrepancy for dense + cross-entropy vs PyTorch was 9.25e-17; one central difference derivative error was 4.21e-12. For convolution, forward and input, weight and bias gradient discrepancies ranged from 5.92e-16 to 2.81e-15 in mean maximum absolute error. The XOR MLP learned all four training points, whereas the perceptron achieved 50%. This is not test accuracy.

Final training cross-entropy at 50 epochs was 0.5369 (SGD), 0.2224 (momentum), and 0.05038 (Adam) for the specified shared learning rate. At initialization scales .01/.5/2, final training loss was .2633/.0791/.1040. These are fixed configurations rather than tuned optimizer comparisons. Held-out accuracy with no regularization was .9000; dropout .3 only .9056; L2 .01 only .9444; both .8833. No general benefit is established. On the noisy-line synthetic task, CNN accuracy was .9917 vs .9583 for an MLP, with differing parameter counts and no independent model-selection test.

## Discussion
Gradient agreement provides evidence within the tested shapes and objectives. ReLU derivative zero on negative inputs is expected, not inherently a defect. The experiment comparing fractions of small derivatives is descriptive. Optimizer effects depend on the chosen learning rate, architecture and budget. The held-out comparisons are exploratory rather than definitive evidence.

## Reproduction
See [README](../README.md), [records](../results/benchmarks.json) and [limitations](limitations.md). Figures are generated from the records by the experiment runner. Original environment and seed metadata are included in the result JSON.

## References
- Goodfellow, Bengio and Courville, Deep Learning (2016), https://www.deeplearningbook.org/
- Kingma and Ba, Adam: A Method for Stochastic Optimization (2014), https://arxiv.org/abs/1412.6980
- PyTorch reference documentation, https://docs.pytorch.org/docs/stable/
- NumPy reference documentation, https://numpy.org/doc/stable/
