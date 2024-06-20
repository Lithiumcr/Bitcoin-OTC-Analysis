# Bitcoin OTC Trust Network Analysis

[![Languages](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-zh-C8161E.svg)](README-zh.md) 
[![GPLv3 licence](https://img.shields.io/badge/license-GPLv3-lightgrey.svg)](LICENSE)

This project aims to analyze the Bitcoin OTC trust network using data mining techniques. The study compares the performance of classical linear regression methods, such as Alternating Least Squares (ALS), with advanced machine learning models, specifically deep learning methods. The dataset used is publicly available from the Stanford Network Analysis Project (SNAP).

## Project Overview

### Objective

The primary goal is to predict trust relationships within the Bitcoin OTC network, balancing accuracy, interpretability, and computational efficiency. This research provides insights into the dynamics of trust and security within decentralized trading environments.

### Methodology

Our approach involves two main methodologies:

1. **Alternating Least Squares (ALS)**
2. **Deep Learning Models** (Model R and Transformers)

### Data Preprocessing

The raw data was normalized to the range of -1 to 1 and split into training (70%), validation (10%), and test (20%) sets. Mean Squared Error (MSE) was used as the evaluation metric.

### Results

- **ALS**: MSE = 0.1250
- **Model R**: MSE = 0.1026
- **Transformers**: MSE = 0.0928

The results indicate that while deep learning models achieve higher accuracy, they are computationally intensive. ALS, on the other hand, is more computationally efficient and suitable for environments with limited resources.

## Detailed Methodologies

### Alternating Least Squares (ALS)

ALS is a classical linear regression method used for matrix factorization in recommendation systems. It projects users and items into a k-dimensional space, approximating the ratings through inner product of latent feature vectors. For details on theoretical derivation, program implementation, and result analysis, please refer to the [project report](./deliverables/ID2211_Project_Group5.pdf) under the deliverables directory.

### Deep Learning Models

Two deep-learning models were employed:

1. **Model R**
2. **Transformers**

These models leverage neural networks to predict link weights with higher accuracy compared to ALS, though at the cost of increased computational requirements.

## Conclusion

The study demonstrates that deep learning models outperform ALS in terms of prediction accuracy. However, ALS remains a viable option for scenarios with limited computational resources due to its efficiency.

## Individual Contributions

- **Changrong Li (李昶融)**: Project coordination, ALS work (literature research, theoretical derivation, development, deployment, result analysis, performance optimisation), report writing (full-text quality control, title, data processing, ALS chapters, appendix).
- **Mānūśrī Tyāgī (मांनुश्री त्यागी)**: Initial proposal frame, ALS work with Changrong Li (development of 3 metrics), summary generation.
- **Zhiqiang Yu (余志强)**: Theoretical research, literature review, validation of deep learning methods.
- **Ziyun Pan (潘梓韫)**: Application of deep learning methods, data preprocessing, and dataset splitting.

## Future Work

Future research will address data imbalances using techniques like SMOTE or GANs, enhance model scalability, and perform a cost-benefit analysis for practical applications.

## Evaluation

> - Jun 17 at 1:34pm
> 
> Good report! The results part could have been improved. It could have been nice to include some more evaluation metrics other than MSE to compare your methods + use confidence intervals (table 3) - Rémi Bourgerie (teaching assistant)
> 
> - Jun 17 at 1:34pm
> 
> Since it is visible from the "Individual Contribution" section that you were the main coordinator and were overseeing all parts of this project, we will give you maximum points for this effort. - Sarunas Girdzijauskas (lecturer, examiner)

## License

This project is licensed under the GPLv3 License. See the LICENSE file for details.
