# Deep learning and Neural Network projects:

A curated collection of **machine learning and data science projects** with a focus on Deep Learning and Neural Networks. 
Covering various areas including **time series forecasting, computer vision, NLP, and predictive analytics**, with a focus on **robust modelling, validation, and real-world applicability**.

---

## Time Series Forecasting

### Electricity Demand Forecasting – Peak Hour Prediction
Predicting whether a given hour will be the **peak electricity demand hour**, framed as a classification / ranking problem rather than raw demand regression.

**Highlights:** temporal feature engineering, seasonality handling, decision-oriented forecasting for energy operations.

---

### Non-Continuous Time Series Inventory Forecasting
Forecasting inventory levels from **irregular and incomplete time series data**, reflecting real-world logging gaps.

**Highlights:** sparse temporal data handling, robust feature construction, stability-focused evaluation.

---

## Computer Vision

### Semi-Supervised Building Segmentation
Segmenting buildings from satellite imagery using **semi-supervised learning**, achieving near-supervised performance with dramatically reduced labelled data.

**Highlights:** data-efficient learning, large-scale geospatial imagery, robustness under limited labels.

---

### Image Pipeline Optimisation for Large-Scale Augmentation
Optimised a TensorFlow image augmentation pipeline, reducing processing time for **100,000 images from 34 minutes to 2 minutes**.

**Highlights:** performance engineering, scalable data pipelines, production-oriented optimisation.

---

### Road Location Detection with GANs
GAN-based model generating road masks from satellite imagery to support infrastructure analysis and road safety scoring.

**Highlights:** generative modelling, image-to-mask tasks, satellite imagery.

---

### Medical Imaging – Cancer Cell Classification & Segmentation
Classification and nuclei segmentation of cancer cells from microscope slides to support downstream diagnostic tasks.

**Highlights:** medical imaging, CNNs, segmentation, data quality and robustness.

---

### Land Use Classification from Satellite Imagery
Classifying **urban vs rural** land use near roads from satellite images to inform infrastructure safety analysis.

**Highlights:** remote sensing, transfer learning, applied classification.

---

### Additional Vision Projects
- Roman numeral classification from handwritten images 
- Scenery classification using transfer learning 
- Synthetic image generation for data-limited ML tasks

---

## Natural Language Processing (NLP)

- Essay grading via semantic text modelling 
- Movie review sentiment classification 
- Working with **high-dimensional embedding spaces** for text representation

---

## Other projects

### Protein Residue–Residue Contact Prediction
Developed a machine learning model to predict **residue–residue contact maps** directly from protein sequences, without explicitly reconstructing 3D structures.

**Highlights:**  
- Built on **ESM2 protein language model embeddings**, extending the contact prediction head  
- Incorporated **structural information from homologous protein sequences** to improve contact prediction accuracy  
- Worked with **high-dimensional embedding representations** and pairwise residue interactions  
- Trained and evaluated using protein structures from PDB, defining contacts via Cα distance thresholds  
- Focused on model interpretability, data preprocessing, and rigorous evaluation rather than end-to-end black-box prediction

**Skills demonstrated:**  
Representation learning, scientific machine learning, embedding-based modelling, pairwise prediction tasks, and integrating domain knowledge into neural architectures.

### Customer Churn Prediction
- Customer return-likelihood prediction from behavioural and event data

---

## Technologies & Skills Demonstrated

- **Programming:** Python, SQL  
- **Time Series:** forecasting, temporal feature engineering, irregular data, leakage-aware validation  
- **Machine Learning:** gradient boosting, deep learning, classification, segmentation  
- **Computer Vision:** CNNs, vision transformers, GANs, satellite & medical imaging  
- **NLP:** text classification, embeddings, semantic evaluation  
- **ML Engineering:** reproducible pipelines, performance optimisation, production-oriented modelling  

---

## Focus

Emphasis on **decision-driven forecasting**, **robust validation**, and **models that generalise under real-world conditions**, rather than optimising for offline metrics alone.
