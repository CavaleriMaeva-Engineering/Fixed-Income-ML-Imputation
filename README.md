**Fixed-Income-ML-Imputation**  
Implementation of a Matrix Factorization engine for High Yield bond spread imputation and Relative Value analysis.

### 1. Presentation

This project was developed during my second year at Télécom SudParis as part of **Project Cassiopée**, a specialized module in Quantitative Finance. The objective of this library is to address the structural illiquidity of the High Yield bond market by using Machine Learning to reconstruct missing OAS (Option-Adjusted Spread) data and build a Relative Value monitoring tool.

### 2. Theoretical Framework

The High Yield market is characterized by sparse trading data, resulting in significant gaps in time series. The engine uses a latent factor approach to estimate "Fair Value" and detect anomalies:

*   **Factorial Spread Model**: Spreads are modeled as a combination of market trends, sector-specific movements, and idiosyncratic risk:  
    $Spread_{i,t} = Base + \beta_i \cdot Market_t + Sector_{s,t} + \epsilon_{i,t}$
*   **Matrix Factorization (SVD)**: The engine utilizes **Singular Value Decomposition** to decompose the sparse spread matrix into latent factors. By reconstructing the matrix, it captures the underlying market structure to fill missing values (NaNs).
*   **Relative Value Strategy**: The model identifies alpha opportunities through Rich/Cheap analysis. If the deviation between the market spread and the SVD-reconstructed Fair Value exceeds a defined threshold, a signal is generated (Buy for undervalued bonds, Sell for overvalued ones).

### 3. Project Structure

The repository is organized following professional modular standards:

*   **src/data_generation.py**: Stochastic simulator creating realistic bond metadata (sectors, ratings) and synthetic spreads with configurable illiquidity rates.
*   **src/models.py**: Implementation of the **OASImputer**, featuring a Naive baseline and an Iterative SVD-based Matrix Factorization algorithm.
*   **src/evaluation.py**: Analytical module for backtesting accuracy via RMSE, segmented by industry sector and credit rating.
*   **src/trading_strategy.py**: Signal generation engine for identifying top Buy/Sell opportunities.
*   **config/settings.yaml**: Centralized configuration for market parameters and model hyperparameters.

### 4. Implementation Details

*   **Language**: Python 3.x
*   **Libraries**: `scikit-learn` (TruncatedSVD) for matrix decomposition, `pandas` & `numpy` for vectorized operations, `matplotlib` & `seaborn` for financial visualization.
*   **Architecture**: Object-Oriented Programming (OOP) to ensure modularity and easy integration of alternative models (e.g., KNN or Deep Learning Autoencoders).

**Career Objective**: Aspiring Quantitative Researcher / Developer. Currently seeking an internship in Quantitative Finance starting in Fall 2026.

**Contact**: Maéva Cavaleri - cavalerimaeva@gmail.com
