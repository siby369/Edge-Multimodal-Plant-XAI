# RESPONSE TO REVIEWER COMMENTS
## ICACRS 2026

**Paper Title:** Multimodal Edge-Optimized Deep Learning with Explainable AI for Plant Disease Detection  
**Authors:** Ms. R. Renugadevi, Divyasri M, Rithikaa K, Siby R  
**Affiliation:** Department of Computer Science and Engineering, KIT - Kalaignarkarunanidhi Institute of Technology, Coimbatore, India  

---

Dear Reviewers,

We sincerely thank the reviewers for their valuable and constructive comments. We have carefully revised the manuscript based on all the suggestions. The revisions improve the technical clarity, experimental reporting, terminology, formatting, and overall language quality of the manuscript. Our point-by-point responses are provided below.

---

### Reviewer Comments 1

**Comment 1:** 1. Dataset information such as sample size, class distribution, preprocessing, and train-test split is not clearly given.  
**Response:** Thank you for the valuable comment. We have added detailed information about the Okra DiseaseNet dataset, including the total sample size, six-class distribution, image preprocessing, normalization, and the stratified training, validation, and testing split. These details are presented in Section III-A and Table I.

**Comment 2:** 2. The proposed ViT-CNN model is not explained in enough technical details to verify.  
**Response:** Thank you for the suggestion. We have expanded the technical description of the proposed hybrid ViT-CNN architecture. The revised manuscript now explains the Depthwise Separable Convolution (DWConv) feature extraction, Swin Transformer blocks, Window Multi-Head Self-Attention (W-MSA), Shifted Window Multi-Head Self-Attention (SW-MSA), multimodal cross-attention fusion, parameter count, and training configuration. These details are provided in Section IV.

**Comment 3:** 3. The quality and usefulness of the C-GAN-generated images are not properly evaluated.  
**Response:** Thank you for pointing this out. We have added quantitative evaluation of the C-GAN-generated images using Fréchet Inception Distance (FID) and Inception Score (IS). In addition, a downstream ablation study has been included to evaluate the usefulness of synthetic samples for disease classification. The results are reported in Section III-B and Table II.

**Comment 4:** 4. The reported inference time is inconsistent, with both below 15 ms and below 50 ms mentioned.  
**Response:** Thank you for identifying this inconsistency. We have clarified the latency according to the target hardware. The revised manuscript reports 14.2 ms on the NVIDIA Jetson Orin Nano for the UAV/edge-GPU profile and 46.8 ms on the ESP32-S3 for the field microcontroller profile. The distinction between the two hardware targets is explicitly explained in Section VI and Table IV.

**Comment 5:** 5. The multimodal approach is discussed, but no clear multimodal experiments or results are presented.  
**Response:** Thank you for the valuable suggestion. We have added a multimodal ablation study comparing telemetry-only, visual-only, and the proposed multimodal fusion pipeline. The revised manuscript reports the corresponding accuracy, precision, and macro F1-score values in Section V-A and Table III.

**Comment 6:** 6. The manuscript requires thorough language editing, including comprehensive proofreading and correction of grammatical, spelling, syntactic, and linguistic errors.  
**Response:** We appreciate this comment. The manuscript has been comprehensively revised and proofread to improve grammar, spelling, punctuation, sentence structure, clarity, and academic language throughout the paper.

**Comment 7:** 7. The authors are strongly advised to have the manuscript carefully reviewed and edited by a fluent English speaker or a professional language-editing service to ensure clarity, readability, and consistency throughout the text. In addition, all tables and figures should be appropriately cited and referred to in the main text using consistent in-text references.  
**Response:** Thank you for the recommendation. The manuscript has been carefully revised for clarity, readability, and consistency. We have also reviewed the captions and in-text references for all tables and figures and ensured that they are appropriately numbered and referred to in the relevant sections of the manuscript.

---

### Reviewer Comments 2

**Comment 1:** 1. Define all the abbreviations at the first case and use them consistently throughout the manuscript.  
**Response:** Thank you for the comment. Abbreviations have been defined at their first occurrence and subsequently used consistently throughout the manuscript. Examples include Convolutional Neural Network (CNN), Vision Transformer (ViT), Conditional Generative Adversarial Network (C-GAN), Depthwise Separable Convolution (DWConv), Explainable Artificial Intelligence (XAI), and Integrated Gradients (IG).

**Comment 2:** 2. Use consistent terminology for Vision Transformer, ViT, Swin Transformer, and hybrid ViT-CNN throughout the manuscript.  
**Response:** Thank you for the suggestion. We have standardized the terminology throughout the revised manuscript. The terms Vision Transformer (ViT), Swin Transformer, and hybrid ViT-CNN are now introduced and used consistently according to their respective architectural roles.

**Comment 3:** 3. Ensure consistent formatting of headings, captions and alignment throughout the manuscript.  
**Response:** Thank you for the comment. We have reviewed and standardized the formatting of section headings, subsection headings, table captions, figure captions, and alignment throughout the revised manuscript.

**Comment 4:** 4. Replace informal phrases with formal sentences throughout the manuscript.  
**Response:** Thank you for the suggestion. Informal and conversational expressions have been revised and replaced with formal academic language throughout the manuscript.

**Comment 5:** 5. The manuscript should be edited for proper English language, grammar, punctuation, spelling, and overall style.  
**Response:** Thank you for the valuable comment. The manuscript has been thoroughly edited to improve English language usage, grammar, punctuation, spelling, sentence construction, and overall academic style.

---

### Conclusion
We sincerely thank the reviewers for their constructive feedback. All comments have been carefully considered and the manuscript has been revised accordingly.
