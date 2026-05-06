# Likely Exam Questions & Answers

Supplement to *Automatic Change-Point Detection in Time Series via Deep Learning*
(Li et al. reimplementation — April 2026)

---

## Core Reformulation

**Q: You say "convert hypothesis testing to binary classification." What exactly do you gain by this reframing?**

The hypothesis testing view requires you to choose a test statistic manually — CUSUM chooses the mean difference, which is optimal only for Gaussian shifts. The classification view lets the model *learn* the optimal test statistic from data via ERM. The DNN implicitly discovers whatever feature best separates "change" from "no-change" for the given noise distribution, without any domain knowledge about the noise.

---

**Q: Why is the class balance fixed at 50/50? Would imbalanced data better reflect reality?**

In real deployments, change points are rare (highly imbalanced). The 50/50 balance is a deliberate training choice to give the model equal exposure to both classes and prevent it from collapsing to "always predict no change." The learned classifier can then be applied with a custom decision threshold at inference time to handle real-world imbalance. This is standard practice in binary detection tasks.

---

## CUSUM

**Q: You state CUSUM gives ~50% accuracy (random guessing) on AR(1) noise. Can you explain why mathematically?**

The CUSUM statistic $T_n = \max_k \sqrt{\frac{k(n-k)}{n}} |\bar X_{1:k} - \bar X_{k+1:n}|$ is derived assuming $\text{Var}(\bar X_{1:k}) = \sigma^2/k$. Under AR(1) with $\rho=0.7$, the actual variance of a sample mean is $\sigma^2/k \cdot \frac{1+\rho}{1-\rho} \approx 5.67\sigma^2/k$. The threshold $\lambda$ calibrated for i.i.d. data is therefore far too permissive — the statistic exceeds $\lambda$ almost always even under $H_0$, inflating false positives and pushing accuracy toward 50%.

---

**Q: Is CUSUM's 50% on S2/S3 a fundamental limit or just poor threshold calibration?**

Partially a threshold issue — one could recalibrate $\lambda$ for each noise type. But for S3 (Cauchy) it is fundamental: Cauchy has no finite mean, so $\bar X_{1:k}$ does not converge to $\mu$ in the usual sense, and no fixed threshold can make the statistic well-behaved. The DNN sidesteps this because it learns rank-based or shape-based features that remain stable under heavy tails.

---

## Theorem 1 & 2

**Q: Theorem 1 says a 1-layer MLP with $2n-2$ ReLU units *exactly* represents CUSUM. What does "exactly" mean here, and why $2n-2$ units specifically?**

"Exactly" means the MLP can compute the same function as CUSUM with zero representation error. The count $2n-2$ comes from needing to compute $n-1$ candidate split points. At each split $k$, a left mean $\bar X_{1:k}$ and a right mean $\bar X_{k+1:n}$ must be computed — each is a piecewise linear function representable by a ReLU unit. Two ReLU units per split point gives $2(n-1) = 2n-2$.

---

**Q: Theorem 2 says the pruned deep MLP uses only $4\lfloor\log_2 n\rfloor$ units per layer. Why does this logarithmic width still work?**

Computing partial means can be done hierarchically. Instead of computing all $n-1$ prefix sums directly (width $\propto n$), a deep network uses a binary-tree reduction: each layer combines adjacent pairs, halving the problem size. $\log_2 n$ layers of binary aggregation suffice to compute any prefix sum, so width $O(\log n)$ with depth $O(\log n)$ replaces width $O(n)$ with depth 1. This is why depth trades width in the generalization bound.

---

## Noise Scenarios

**Q: Why specifically AR(1) with $\rho = 0.7$ for S1'? What makes 0.7 a meaningful choice?**

$\rho = 0.7$ is a moderate-to-strong autocorrelation — strong enough to substantially inflate the variance of sample means (inflation factor $\frac{1+0.7}{1-0.7} \approx 5.7\times$), but still stationary ($|\rho| < 1$). It is a value commonly used in econometrics and signal processing benchmarks to represent "typical" persistent time series. Values close to 1 would approach a unit root (non-stationary), which is a different problem entirely.

---

**Q: Why Cauchy for S3 and not Student-t with small degrees of freedom?**

Cauchy is the extreme case of heavy tails — it has no defined mean or variance, making it the hardest possible case for mean-based statistics like CUSUM. Student-t with even 3 degrees of freedom has a defined mean, so CUSUM degrades gracefully. Cauchy is used to demonstrate the *breakdown* of classical methods, not just degradation. It also corresponds to a Student-t with 1 degree of freedom, connecting it to a well-known family.

---

## Data Augmentation

**Q: You reverse sequences to double training size. Is this augmentation theoretically valid — does a reversed sequence have the same labeling semantics?**

The answer depends on the scenario. For **S1** (i.i.d. Gaussian) and **S3** (Cauchy), the noise is i.i.d., so reversal produces a sequence from exactly the same distribution — the augmentation is unambiguously valid. For **S1'** (AR(1) with fixed $\rho$), stationary AR(1) Gaussian processes are time-reversible, so the reversed sequence is also a valid AR(1) draw. For **S2** (time-varying $\rho_t$), the forward process $x_t = \rho_t x_{t-1} + \varepsilon_t$ is causal; its reversal produces an anti-causal process $x_t = \rho_{n-t} x_{t+1} + \varepsilon_{n-t}$, which is a different conditional structure. However, the binary label (change present / not present) is preserved, and the reversed sequence still contains the same mean-shift signal. In practice the augmentation adds distributional diversity and helps training, but it is not theoretically clean for S2.

---

## Normalization

**Q: Why min-max for S1/S1'/S2 but "trimmed robust" normalization for S3?**

Min-max normalization divides by the range $(\max - \min)$. For Cauchy sequences, extreme outliers make the range arbitrarily large, effectively squashing all signal values toward zero. Trimmed robust normalization (e.g., using the 5th–95th percentile range) is resistant to those outliers and preserves the relative structure of the sequence. Applying min-max on Cauchy data would collapse the network's input near zero, causing the model to fail despite good architecture.

---

## ResCNN Architecture

**Q: Why 21 residual blocks specifically? How sensitive is performance to this depth?**

The 21-block depth is adopted from the original paper's architecture for time-series classification. With kernel size 3 per block, the effective receptive field is approximately $2 \times 21 + 1 = 43$ samples, covering nearly half a sequence of length $n = 100$. We did not perform an ablation on depth — this is acknowledged as a limitation in the slides.

---

**Q: Skip connections prevent vanishing gradients — but you also use dropout 0.3. Isn't dropout counterproductive in a residual network?**

Not necessarily. Dropout is applied *within* the residual branch (before the skip addition), not on the skip connection itself. The skip path always propagates gradients cleanly. The dropout on the residual branch acts as regularization to prevent co-adaptation of features, which is particularly useful for the small synthetic datasets ($N = 10{,}000$). This is consistent with how dropout is used in WideResNet variants.

---

**Q: Why does ResCNN outperform MLP so dramatically on Cauchy noise (+44 pp)?**

Cauchy has infinite variance; global statistics like sample means are meaningless. The CNN's local receptive fields detect *local* shape changes (sudden shifts in local amplitude or pattern) rather than global mean differences. Skip connections preserve these local features across depth without saturation. The MLP, operating on the full flattened sequence, attempts to fit global statistics that are undefined for Cauchy data — so it cannot learn a reliable decision boundary.

---

## HASC Dataset

**Q: You say "no architectural redesign" to go from binary to 30-class. What exactly changed?**

Only the output layer: the binary sigmoid head (1 unit) was replaced by a 30-class softmax head (30 units), and the loss changed from BCE to categorical cross-entropy. All 21 residual blocks — the entire feature extraction backbone — are unchanged. The learned hierarchical local features generalize from synthetic binary classification to real multi-class sensor data.

---

**Q: The footnote says "current split metadata may reflect later reruns." Does this undermine the 96.21% result?**

It means we cannot guarantee person-level train/test separation, which is the gold standard for generalization on HASC. If the same person appears in both train and test, the model may be memorizing individual gait patterns rather than generalizing to new people. The result is real but should be interpreted as an upper bound on true generalization. Person-held-out splits are listed as future work precisely for this reason.

---

## Conclusion

**Q: You conclude "deep learning adapts where classical methods break down." But DNNs are also data-hungry — isn't that a different kind of breakdown?**

This is a valid tension. Theorem 1 requires $N \gg n^2 \log n$ training samples — for $n = 100$ that means $\gg 10^5$ samples, yet we use only 10,000. The bound is loose and the model works in practice, but in truly data-scarce regimes CUSUM's closed-form nature is an advantage. The claim holds when labeled data is cheap to generate synthetically, but it is not a universal statement.

---

**Q: Only CUSUM is compared as a baseline — why not PELT, E-divisive, or BOCPD?**

This is acknowledged as a limitation. PELT and E-divisive are non-parametric methods that do not assume Gaussian noise, so they would be fairer comparisons on S2 and S3. BOCPD provides probabilistic uncertainty estimates, which the current binary classifier does not. Extending the comparison is the most pressing future work item to strengthen the empirical claims.

---

**Q: Algorithm 1 (multi-CP localization) is "demonstrated qualitatively only." What would a proper quantitative evaluation look like?**

A proper evaluation would generate synthetic sequences with $m \geq 2$ known change points, run Algorithm 1, and report: (1) the mean absolute localization error $\mathbb{E}[|\hat\tau - \tau|]$, (2) precision/recall on detected vs. true change points, and (3) the false discovery rate as a function of threshold $\gamma$. This would replace the current single-example demonstration with statistically meaningful numbers.

---

## Deep Conceptual Challenges

**Q: The title says "automatic" detection. But you train a separate model per noise scenario (S1, S1', S2, S3). Doesn't the user still need to know the noise type?**

This is a real limitation of the experimental setup. There are four separate trained models — one per scenario — confirmed by the config files (`mlp_s1.yaml`, `mlp_s2.yaml`, etc.). In deployment, a user would need to select the right model for their noise type, which requires prior knowledge. The word "automatic" refers to *feature engineering* automation (no hand-crafted test statistic needed), not to model selection automation. A truly automatic system would either train a single model across all noise types or use a noise-type classifier first. This is an honest gap between the paper's framing and the actual implementation.

---

**Q: CUSUM is asymptotically minimax-optimal under Gaussian i.i.d. noise, yet your S1 result shows ResCNN at 0.9255 vs CUSUM at 0.8915. How can the DNN beat an optimal method?**

It cannot beat an *oracle* CUSUM calibrated perfectly for finite $n$. The CUSUM threshold $\lambda$ in the experiments is calibrated using asymptotic tables designed for $n \to \infty$. At $n = 100$, the finite-sample threshold is slightly off, causing either excess false positives or false negatives. The DNN, trained end-to-end on the same $n = 100$ distribution, implicitly learns the correct finite-sample decision boundary. So the gap is not "DNN > optimal CUSUM" — it is "DNN > textbook asymptotic CUSUM at finite $n$." The asymptotic optimality of CUSUM is not violated.

---

**Q: The generalization bounds in Theorems 1 and 2 are for the Empirical Risk Minimizer $h_{\text{ERM}}$. But SGD on a non-convex loss does not find the global ERM. Do the bounds actually apply to your trained model?**

Strictly, no. The theorems guarantee risk bounds for the model that *exactly minimizes* the training loss. SGD finds a local minimum, which may have higher training loss than the global ERM. The gap between the theoretical guarantee and the trained model is an open problem in deep learning theory. In practice, for overparameterized networks trained with early stopping and weight decay, the local minima found by SGD have been empirically shown to generalize well — but this is not covered by the theorem. You should present the theoretical results as motivation and upper bounds, not as tight guarantees on your specific trained checkpoints.

---

**Q: What does the $\sqrt{\eta(1-\eta)}$ factor in the SNR definition mean geometrically?**

$\eta = \tau/n$ is the relative position of the change point. The factor $\eta(1-\eta)$ is maximized at $\eta = 0.5$ (change in the middle of the sequence) and approaches 0 as $\eta \to 0$ or $\eta \to 1$ (change near the boundaries). Geometrically, it measures how much evidence is available on both sides of the split: a central change gives the most balanced evidence ($n/2$ samples on each side), while a boundary change gives almost no evidence on one side. Consequently, the class $\Theta(B)$ excludes near-boundary change points unless the mean jump $|\mu_L - \mu_R|$ is very large — the theory does not cover changes in the first or last few samples.

---

**Q: Theorem 1 uses "shifted ReLU $\sigma_b$" rather than standard ReLU. Why does the shift matter for representing CUSUM?**

The CUSUM statistic requires computing partial sums $\sum_{t=1}^k x_t$ for each candidate split $k$. In a neural network, a standard ReLU $\max(0, x)$ fires for any positive input. A shifted ReLU $\max(0, x - b)$ fires only when $x > b$, acting as a threshold gate. This is essential for the MLP to select *which* inputs contribute to the partial sum at split $k$ — the shift $b$ encodes the position $k$ implicitly. Without the shift, the network cannot represent distinct split-point statistics in a single hidden layer.

---

**Q: Your sample complexity bound for the shallow MLP requires $N \gg n^2 \log n$, but for $n=100$ that means $N \gg 46{,}000$, yet you use only $N = 10{,}000$. Aren't you violating your own theoretical requirements?**

Yes, by the letter of Theorem 1, $N = 10{,}000$ is insufficient. Three points in response: (1) the bound is a worst-case upper bound — the constant $C$ is large and the bound is not tight in practice; (2) Theorem 2 (the deep pruned MLP) has a much better bound, $N \gg n\log^2 n \approx 4{,}600$ for $n=100$, which $N=10{,}000$ satisfies; (3) the empirical results validate that the model generalizes despite the shallow bound not being met. This is a common situation in deep learning: practice outperforms worst-case theory. The pruned deep MLP is the architecturally correct choice, and its bound is consistent with the experiment.

---

**Q: Macro F1 is 0.9042 but Weighted F1 is 0.9608 on HASC. What does the 0.057 gap tell you?**

Macro F1 gives equal weight to every class regardless of sample count; weighted F1 weights by class support (number of samples). The gap of 5.7 pp indicates that rare classes — specifically transition activities with fewer than 10 validation samples — have substantially lower F1 than the common pure-activity classes. The high weighted F1 is flattering because it is dominated by the large pure-activity classes where the model is nearly perfect. The lower macro F1 reveals that the model struggles on rare transition states, which are precisely the most informative events for change-point detection. A balanced evaluation should report both.

---

**Q: The HASC experiment is 30-class activity classification, not change-point detection. Why is it in a change-point detection paper?**

Activity transitions *are* change points. The 30 classes include 6 pure activities and 24 transition activities (e.g., walk$\to$jog, stand$\to$walk). A window classified as a transition class implicitly identifies that a behavioral change is occurring within that window — this is the change-point detection problem in a real-world disguise. The experiment demonstrates that the same architecture that detects distributional shifts in synthetic data can also recognize and label the *type* of transition, which is a strictly harder task than binary detection.

---

**Q: Accuracy is the only metric for Experiment 1. In real monitoring applications, is accuracy the right criterion?**

Usually not. In anomaly monitoring, false negatives (missed changes) are typically far more costly than false positives (false alarms). Accuracy treats both error types equally. More informative metrics would be: precision and recall separately, the F1 score, or the ROC-AUC which shows the tradeoff across all thresholds. Reporting only accuracy at a fixed 0.5 threshold hides the model's operating characteristics. A practitioner deploying this system would tune the threshold based on the cost ratio of false negatives to false positives, not optimize for accuracy.

---

**Q: What does the DNN actually learn on S3 (Cauchy noise)? You claim it picks up "shape-based features" — can you verify this?**

Honestly, no — not without interpretability analysis such as gradient saliency maps, SHAP values, or activation visualization. The claim that the CNN detects local shape changes rather than global means is a reasonable hypothesis (CNN receptive fields are local; Cauchy means are undefined), but it is not verified in the paper or our reimplementation. This is an open interpretability question. A rigorous answer would require analyzing which input positions maximally activate the neurons that drive the change/no-change decision for Cauchy sequences.

---

## Simple Baseline Questions

**Q: What exactly is a change point — give an intuitive example.**

A moment where the underlying statistics of a time series shift abruptly. For example, a patient's heart rate before and after a drug is administered, or stock return volatility before and after a market shock. Formally, the distribution generating $X_t$ changes at some index $\tau$.

---

**Q: What is the difference between detection and localization?**

Detection is binary: does any change exist in the sequence? Localization asks: at which index $\hat\tau$ does it occur? Experiment 1 is detection — the model outputs 0 or 1. Algorithm 1 is localization — it additionally reports the estimated position of each change point within a long series.

---

**Q: What does ERM mean?**

Empirical Risk Minimization. Find the model parameters $\theta$ that minimize the average loss on the training set: $\hat\theta = \arg\min_\theta \frac{1}{N}\sum_{i=1}^N \mathcal{L}(h_\theta(x_i), y_i)$. It is the theoretical framework that the generalization bounds in Theorems 1 and 2 are built around.

---

**Q: Why Adam over plain SGD?**

Adam maintains per-parameter adaptive learning rates using first moment ($\beta_1 = 0.9$) and second moment ($\beta_2 = 0.999$) estimates of the gradient. This handles sparse or noisy gradients well and converges faster without careful manual tuning of the learning rate schedule. Plain SGD uses a single global learning rate that requires extensive tuning to match Adam's performance.

---

**Q: What does "BCE with logits" mean? Why not compute sigmoid first, then BCE?**

BCE with logits feeds the raw pre-sigmoid score directly into a numerically stable combined formula. Computing sigmoid then BCE separately risks $\log(0)$ when the logit is very large (sigmoid saturates to exactly 1) or very small (sigmoid saturates to exactly 0), producing NaN. The fused implementation avoids this by working in log-space throughout.

---

**Q: What does early stopping actually do — what triggers it?**

Each epoch, validation loss is recorded. If it does not improve for `patience = 20` consecutive epochs, training halts and the best checkpoint (lowest validation loss seen so far) is restored. This prevents overfitting without requiring a fixed epoch budget and is cheaper than full cross-validation.

---

**Q: What is the validation set used for here?**

Hyperparameter selection and early stopping — weights are never updated using validation data. It acts as a proxy for generalization: if validation loss rises while training loss falls, the model is overfitting. It is held out from the test set, which is used only for final reporting.

---

**Q: What does "deterministic dataset with SHA-256 hashes" mean practically?**

The synthetic datasets are generated once and saved to disk. A SHA-256 hash of each file is recorded at generation time. Before every training run, the hash is recomputed and compared. If it differs, the data was modified or regenerated with a different seed — breaking reproducibility. This ensures all reported results come from exactly the same data splits.

---

**Q: What is a residual connection — describe it in one sentence.**

The output of a block is $y = \mathcal{F}(x) + x$: the raw input $x$ is added directly to the transformed output $\mathcal{F}(x)$, so gradients can flow backward through the identity shortcut without passing through any nonlinearity, preventing vanishing gradients in deep networks.

---

**Q: What does dropout do and when is it active?**

During training, dropout randomly zeroes each activation with probability $p = 0.3$, forcing the network to learn redundant representations rather than relying on any single neuron. At inference time dropout is disabled and activations are scaled by $(1 - p)$ to keep the expected output magnitude consistent.

---

**Q: What is softmax and why use it for HASC instead of sigmoid?**

Softmax normalizes raw logits across all $K = 30$ classes into a probability distribution that sums to 1, enforcing mutual exclusivity — exactly one activity per window. Sigmoid would produce independent probabilities per class, allowing the model to simultaneously assign high confidence to multiple classes, which is semantically wrong for single-label classification.

---

**Q: What does a confusion matrix show?**

Rows are true classes, columns are predicted classes. Diagonal entries are correct predictions; off-diagonal entries show which classes are confused with which. For HASC, it reveals whether rare transition activities are mislabeled as their adjacent pure activities — which is the most likely failure mode given class imbalance.

---

**Q: Why does your training accuracy not appear in the results table?**

Reporting only test accuracy follows the standard convention for evaluating generalization. Training accuracy is expected to be high (the model fits its own data); what matters is performance on unseen data. Including training accuracy without a generalization gap analysis would add little information and might mislead readers into thinking high training accuracy is the claim.

---

**Q: What is the difference between macro and weighted F1 — which is more honest here?**

Macro F1 averages per-class F1 scores with equal weight regardless of class size. Weighted F1 weights each class by its number of samples. For HASC with 30 heavily imbalanced classes (some transitions have fewer than 10 samples), macro F1 is more honest: it does not let the large pure-activity classes — where the model is nearly perfect — mask poor performance on rare transitions, which are the most informative events for change-point detection.
