# LLMrec Large Language Model Recommendation 

LLMrec is a recommendation system that uses large language models to augment the training data of a collaborative filtering model. This documentation documents the API of the LLMrec package.See also

## Table des Matières

- [Aperçu](#aperçu)
- [Authors](#authors)
- [Get the package](#get-the-package)
- [Content](#content)
- [Our optmiztion](#our-optimizations)
- [Our Contributions](#our-contributions)
- [Api documentation](#api-documentation)
- [Citing](#citing)

## Aperçu

The repo is a reimplementation of the official LLMRec project. This enhanced version incorporates various code and implementation improvements, along with an extended test suite to ensure robustness and reliability.

## Authors

This project was developed by a team of passionate contributors :

- **[DRIOUA Imed](https://github.com/imedDrioua)**:  Data scientist and a student at University of Paris.
- **[BOUFAFA Lamis](https://github.com/Lamis1847)**: Data scientist and a student at University of Paris.

## Get the package
Use the following git command to clone the repository : <br>

```git clone https://github.com/imedDrioua/LLM_REC.git```

## Content

The repository contains three branches, each dedicated to different datasets:

- **Main** : Implementation for the Amazon Books dataset (Master branch).
- **Netflix** : Implementation for the Netflix dataset.
- **amazon_books** : Implementation for the Amazon dataset (For testing purposes to keep the master branch clean).

To run the code, execute the following command in your terminal:

```bash
python main.py
```

## Original Repository
The original repository can be found [here](https://github.com/HKUDS/LLMRec).
## Our Optimizations

We have implemented the LLMRec model on the Netflix dataset and introduced several enhancements to the original code.

Our code optimizations include:

- **Project Structuring as a Package**: Organizing the project into a structured package format for better management and scalability.
- **Data Source Centralization in a Single Class**: Centralizing data sources within a single class to streamline data management and access.
- **Implementation of Models Following PyTorch Standards**: Adhering to PyTorch standards for model implementation to ensure compatibility and maintainability.
- **Introduction of Exploratory Notebooks**: Providing exploratory notebooks to facilitate understanding and experimentation with the codebase.
- **Elaboration of Comprehensive Project Documentation**: Creating detailed project documentation to aid in understanding, usage, and contribution.

Our algorithmic optimizations include:

- **Reimplementation of LightGCN (Following its official paper)**: Reimplementing LightGCN based on its official paper to improve performance and accuracy.
- **Reimplementation of BPR Loss Function (Following LightGCN official paper)**: Implementing the Bayesian Personalized Ranking loss function according to the LightGCN official paper for better training convergence.
- **Stabilization of Training with Dropout and Batch Normalization**: Incorporating dropout and batch normalization techniques to stabilize training and prevent overfitting.
- **Aggregation of Augmented Data Embeddings**: Aggregating embeddings generated from augmented data sources to enhance items attributes representation.

## Our Contributions

We have enriched the Amazon Books dataset with augmented data using:

- **Cohere LLM for Data Augmentation**: Leveraging Cohere LLM for augmenting the Amazon Books dataset with additional context and diversity.
- **LangChain for Prompting**: Utilizing LangChain for generating prompts to enrich the dataset with diverse textual inputs.
- **SentenceBert and ViT for Embedding Generation**: Employing SentenceBert and ViT for generating embeddings to capture semantic relationships and improve recommendation quality.
## API Documentation
The docs folder contains the API documentation for the LLMrec package. The documentation provides detailed information about the package's classes, methods, and attributes, along with examples to illustrate their usage.

## Citing

This work is based on the following paper. If you use this code, please consider citing it.

```
@article{wei2023llmrec,
  title={LLMRec: Large Language Models with Graph Augmentation for Recommendation},
  author={Wei, Wei and Ren, Xubin and Tang, Jiabin and Wang, Qinyong and Su, Lixin and Cheng, Suqi and Wang, Junfeng and Yin, Dawei and Huang, Chao},
  journal={arXiv preprint arXiv:2311.00423},
  year={2023}
}
```