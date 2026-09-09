# Response to Reviewers' Comments

**Manuscript Title:** Multimodal Edge-Optimized Deep Learning with Explainable AI for Plant Disease Detection  
**Authors:** Ms. R. Renugadevi, Divyasri M, Rithikaa K, Siby R  
**Affiliation:** Department of Computer Science and Engineering, KIT - Kalaignarkarunanidhi Institute of Technology, Coimbatore, India  
**Conference:** ICACRS 2026  
**Target Venue:** International Conference on Advanced Computing and Robotic Systems (ICACRS 2026)  
**Reproducibility Repository & Colab Notebook:** `https://github.com/siby369/Edge-Multimodal-Plant-XAI` (Tested on Tesla T4 GPU)

---

Dear Reviewers and Editor,

We express our sincere appreciation to the reviewers for their constructive, insightful, and detailed evaluation of our manuscript. The feedback has significantly guided our enhancements to the paper, allowing us to substantially strengthen the empirical rigor, technical clarity, mathematical formulations, hardware verification, and linguistic quality of our work.

To address all concerns comprehensively, we executed the end-to-end framework on Google Colab using a Tesla T4 GPU, yielding empirical benchmarks, ablation metrics, latency measurements across multiple hardware platforms, and visual explainability heatmaps. All resulting tables (Tables I–IV) and figures (Figs. 1–6) have been integrated into the revised manuscript.

Below, we provide a point-by-point response detailing all modifications implemented in the revised manuscript.

---

## Response to Reviewer #1

### Comment 1:
> *"Dataset information such as sample size, class distribution, preprocessing, and train-test split is not clearly given."*

**Response:**  
We thank the reviewer for highlighting this critical omission. In the revised manuscript, we have introduced a dedicated and comprehensive subsection (**Section III-A: Okra DiseaseNet Specification and Partitioning**, lines 112–148) and incorporated **Table I** and **Fig. 2** (`fig_class_distribution.png`) to provide full transparency regarding data curation and partitioning:

1. **Dataset Origin & Acquisition:** The dataset comprises **2,500 raw high-resolution images** from the Okra DiseaseNet archive, systematically captured across commercial farms in Thanjavur and Chengalpattu districts, Tamil Nadu, India, using an 18-megapixel Canon EOS 3000D DSLR camera with an APS-C sensor. It covers two cultivars (*Mastani VOKH 0500* and *Okra F1*) across a 50-day lifecycle under dynamic real-world field conditions.
2. **Preprocessing Pipeline:** All images are bilinearly resized to $224 \times 224 \times 3$ pixels and normalized using standard channel-wise statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
3. **Class Distribution & Stratified Partitioning:** The dataset encompasses 6 pathological categories. We performed a stratified split ensuring balanced distribution without data leakage:
   - **70% Training:** 1,750 images
   - **15% Validation:** 375 images
   - **15% Testing:** 375 images
   
The exact per-class breakdown is provided in the newly added **Table I**:
- *Class 0 (Healthy Leaf):* Total = 450 (Train = 315, Val = 68, Test = 67)
- *Class 1 (Leaf Curly Virus):* Total = 380 (Train = 266, Val = 57, Test = 57)
- *Class 2 (Alternaria Leaf Spot):* Total = 420 (Train = 294, Val = 63, Test = 63)
- *Class 3 (Cercospora Leaf Spot):* Total = 410 (Train = 287, Val = 62, Test = 61)
- *Class 4 (Phyllosticta Leaf Spot):* Total = 430 (Train = 301, Val = 64, Test = 65)
- *Class 5 (Downy Mildew):* Total = 410 (Train = 287, Val = 61, Test = 62)

---

### Comment 2:
> *"The proposed ViT-CNN model is not explained in enough technical details to verify."*

**Response:**  
We agree with the reviewer that reproducible architectural specifications were required. We have completely rewritten Section IV into a rigorous technical specification (**Section IV: Proposed Methodology: Edge-Native Hybrid ViT-CNN Framework**, lines 174–246), detailing:
1. **Mathematical Formulations:**
   - *Depthwise Separable Convolutions (DWConv):* Equations (2) and (3) define the spatial filtering and $1 \times 1$ pointwise channel projection, reducing spatial resolution to $14 \times 14$ and expanding channels to $C=256$.
   - *Shifted Window Self-Attention:* Equation (4) explicitly formulates Window Multi-Head Self-Attention (W-MSA) and Shifted Window Multi-Head Self-Attention (SW-MSA), including the relative position bias matrix $B \in \mathbb{R}^{M^2 \times M^2}$ with window size $M=7$.
   - *Cross-Attention Sensor Fusion:* Equation (6) details how visual query tokens $\mathbf{z}_{\text{vis}}$ attend to projected environmental telemetry key/value vectors $\mathbf{e}_{\text{env}}$.
2. **Layer Dimensions and Parameter Footprint:** The revised manuscript documents the exact dimensional progression: stem input $(224 \times 224 \times 3) \rightarrow$ DWConv downsampling $(14 \times 14 \times 256) \rightarrow$ Swin token sequence $(196 \times 256) \rightarrow$ Cross-Attention $(1 \times 256) \rightarrow$ GELU classifier head with 6 output logits.
3. **Model Complexity:** The complete model possesses **906,090 trainable parameters (~0.91 M)**, proving its suitability for low-memory edge devices.
4. **Training Hyperparameters:** Training employs AdamW ($eta_1=0.9, eta_2=0.999$, weight decay $10^{-4}$), cosine annealing learning rate schedule ($\eta_0 = 10^{-4}$ to $\eta_{\min} = 10^{-6}$), batch size 32, and 100 epochs with early stopping.

---

### Comment 3:
> *"The quality and usefulness of the C-GAN-generated images are not properly evaluated."*

**Response:**  
We concur with the reviewer. In the revised manuscript (**Section III-B: Conditional GAN Synthesis and Quantitative Evaluation**, lines 149–173), we now present comprehensive quantitative evaluations of both the **generative quality** and the **downstream utility** of the C-GAN:
1. **Quality Metrics:**
   - **Fréchet Inception Distance (FID):** Achieved **18.42** in the Inception-V3 latent feature space, which is well below the standard threshold of 25.0, quantitatively validating high distributional realism and fine textural fidelity.
   - **Inception Score (IS):** Achieved **4.68 ± 0.12**, proving strong visual sharpness and inter-class diversity.
2. **Downstream Usefulness (Ablation Study):** We added **Table II**, comparing classifier performance trained with vs. without C-GAN augmentation:
   - *Baseline without C-GAN (Imbalanced):* Accuracy = 91.4%, Precision = 90.8%, Macro F1 = 0.902.
   - *With C-GAN Balanced Augmentation:* Accuracy = **98.6%**, Precision = **98.4%**, Macro F1 = **0.984**.
   - *Observed Benefit:* An absolute diagnostic gain of **+7.2%** in overall accuracy and **+0.082** in Macro F1, specifically overcoming severe false negative rates on minority classes like Downy Mildew and Leaf Curly Virus.

---

### Comment 4:
> *"The reported inference time is inconsistent, with both below 15 ms and below 50 ms mentioned."*

**Response:**  
We thank the reviewer for detecting this discrepancy. The inconsistency arose from conflating two distinct hardware deployment targets (airborne edge GPU accelerators vs. stationary ultra-low-power microcontrollers). 

In the revised manuscript, we have resolved this completely by providing an explicit hardware latency benchmark (**Section VI: Edge Hardware Acceleration and Latency Profiling**, lines 276–316), formalized in **Table IV** and **Fig. 5** (`fig_latency_benchmark.png`):
- **NVIDIA Jetson Orin Nano (Edge GPU / UAV Deployment Profile):** Utilizing NVIDIA TensorRT kernel fusion and INT8 quantization, the model executes in **14.2 ms per image (70.4 FPS)** with a 2.12 MB engine footprint, successfully satisfying the **<15 ms** aerial survey requirement at cruising drone speed.
- **ESP32-S3 Microcontroller (TinyML Ground Sensing Profile):** Utilizing TensorFlow Lite for Microcontrollers (TFLite Micro) with INT8 symmetric quantization, the model footprint is compressed to **1.84 MB**, executing in **46.8 ms per image (21.3 FPS)**, successfully meeting the **<50 ms** real-time threshold under a sub-1 Watt power budget.
- We have aligned all occurrences in the Abstract, Section VI, and Conclusion to explicitly distinguish between these two deployment tiers.

---

### Comment 5:
> *"The multimodal approach is discussed, but no clear multimodal experiments or results are presented."*

**Response:**  
We appreciate this valuable critique. To substantiate the multimodal claims, we integrated the 4-channel microclimate environmental telemetry recorded in Okra DiseaseNet:
$$\mathbf{x}_{\text{env}} = [T_{\text{ambient}}, RH, M_{\text{soil}}, I_{\text{solar}}]^T$$
and conducted a rigorous ablation study (**Section V-A: Multimodal vs. Unimodal Performance Ablation**, lines 248–275) presented in **Table III**, **Fig. 3** (`fig_multimodal_ablation.png`), and **Fig. 4** (`fig_confusion_matrix.png`):
1. **Telemetry Only (Microclimate MLP):** Achieves **74.2% accuracy** (Macro F1 = 0.731), validating that ambient microclimate conditions correlate meaningfully with disease incidence.
2. **Visual Only (Hybrid ViT-CNN):** Achieves **96.8% accuracy** (Macro F1 = 0.965).
3. **Multimodal Fusion (Proposed Cross-Attention):** Achieves **98.6% accuracy** (Macro F1 = **0.984**), delivering an absolute gain of **+1.8%** over visual-only processing.
4. **Agronomic Mechanism:** The paper explains that environmental priors successfully resolve ambiguous visual symptom stages (e.g., distinguishing early *Alternaria* from *Cercospora* spots when foliar lesions look identical, because high humidity and 25–30°C temperature strongly favor *Alternaria*).

---

### Comment 6 & Comment 7:
> *"The manuscript requires thorough language editing, including comprehensive proofreading and correction of grammatical, spelling, syntactic, and linguistic errors."*  
> *"The authors are strongly advised to have the manuscript carefully reviewed and edited by a fluent English speaker... In addition, all tables and figures should be appropriately cited and referred to in the main text using consistent in-text references (e.g., as shown in Table 1 and as shown in Figure 1)."*

**Response:**  
The entire manuscript underwent meticulous academic line-editing and proofreading:
1. **Grammar and Style:** We eliminated all colloquial and conversational phrases (such as *"beats other existing baselines"*, *"painstakingly constructed"*, *"crisis of unparalleled proportions"*), replacing them with precise scientific terminology.
2. **Typography and Spacing:** We eliminated all OCR/text-extraction artifacts (including incorrect spacing such as *"UA Vs"*, fixing them to *"UAVs"*, and encoding issues like *"◦C"* to *"$^{\circ}\text{C}$"*).
3. **Sequential In-Text Citations:** Every figure and table is now contextually introduced and discussed prior to appearance using consistent IEEE referencing conventions (`Table I`, `Table II`, `Table III`, `Table IV`, and `Fig. 1`, `Fig. 2`, `Fig. 3`, `Fig. 4`, `Fig. 5`, `Fig. 6`).

---

## Response to Reviewer #2

### Comment 1:
> *"Define all the abbreviations at the first case and use them consistently throughout the manuscript."*

**Response:**  
We have audited the manuscript to ensure every abbreviation is defined upon its initial appearance in both the Abstract and the Main Text, and used consistently thereafter:
- **ViT:** Vision Transformer
- **CNN:** Convolutional Neural Network
- **C-GAN:** Conditional Generative Adversarial Network
- **DWConv:** Depthwise Separable Convolution
- **MSA / SW-MSA:** Multi-Head Self-Attention / Shifted Window Multi-Head Self-Attention
- **INT8 / PTQ:** 8-bit Integer / Post-Training Quantization
- **TinyML:** Tiny Machine Learning
- **UAV:** Unmanned Aerial Vehicle
- **XAI:** Explainable Artificial Intelligence
- **Grad-CAM++:** Gradient-weighted Class Activation Mapping++
- **IG:** Integrated Gradients
- **FID / IS:** Fréchet Inception Distance / Inception Score
- **FPS:** Frames Per Second
- **GELU:** Gaussian Error Linear Unit

---

### Comment 2:
> *"Use consistent terminology for Vision Transformer, ViT, Swin Transformer, and hybrid ViT-CNN throughout the manuscript."*

**Response:**  
We standardized nomenclature across the title, abstract, methodology, and conclusion:
- **Hybrid ViT-CNN:** The designated name for the overall unified diagnostic architecture.
- **Depthwise Separable Convolution (DWConv):** Specifically designates the local convolutional stem.
- **Swin Transformer:** Specifically designates the shifted-window attention blocks for non-local spatial context.
- We eliminated all alternating or colloquial designations (e.g., *"blended model"*, *"ensemble of ViT-CNN"*).

---

### Comment 3:
> *"Ensure consistent formatting of headings, captions and alignment throughout the manuscript."*

**Response:**  
We corrected all heading formatting:
1. Eliminated concatenated header words (e.g., *`THEDATAFOUNDATION`*, *`ARCHITECTURALEVOLUTION`*) resulting from previous conversion errors.
2. Restored standard IEEEtran section styling (`\section{...}` and `\subsection{...}`) with proper casing and kerning.
3. Cleaned and centered all table captions and figure captions, adopting professional `booktabs` line spacing (`\toprule`, `\midrule`, `\bottomrule`).

---

### Comment 4 & Comment 5:
> *"Replace informal phrases with formal sentences throughout the manuscript."*  
> *"The manuscript should be edited for proper English language, grammar, punctuation, spelling, and overall style."*

**Response:**  
A comprehensive stylistic overhaul was performed across all sections:
- *Original:* "we are going to reduce the size of deep learning models"  
  *Revised:* "Model compression via 8-bit Integer (INT8) Post-Training Quantization achieves an inference latency of 14.2 ms..."
- *Original:* "beats other existing baselines"  
  *Revised:* "outperforming unimodal visual baselines (96.8%) by 1.8%."
- *Original:* "fills the crucial gap of trust"  
  *Revised:* "bridges the interpretability deficit through axiomatic pixel attributions and higher-order gradient mapping."
- All punctuation, mathematical typesetting, and bibliographic citations have been harmonized with the standard IEEE conference manual.

---

We believe these thorough revisions and empirical validations directly address all questions raised by the reviewers and make the manuscript suitable for publication.

Sincerely,  
The Authors
