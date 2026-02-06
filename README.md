# New Repository Game Predictor 

**Project for Master 2 – Predict GitHub Repository Creation Trends**

This project aims to use the GitHub API to **predict the number of new repositories created weekly** for a specified game. It uses machine learning models that are retrained weekly, automatically choosing the best model for prediction based on past data.

---

## Features

- **Weekly Prediction**  
  Predicts how many new GitHub repositories will be created in the next week for a specified game. :contentReference[oaicite:2]{index=2}

- **Multiple Machine Learning Models**  
  Retrains and compares several models weekly, then selects the best model to use for predictions. :contentReference[oaicite:3]{index=3}

- **Automated Training & Data Collection**  
  Entire pipeline from data collection to retraining and prediction is automated. :contentReference[oaicite:4]{index=4}

-  **API Endpoints Available**  
  Includes APIs for:
  -  Manual training
  -  Data collection
  -  Making predictions :

---

## Tech Stack
-  **Python** – core logic and models :contentReference[oaicite:6]{index=6}  
-  **Docker** & **docker-compose** – containerization and setup :contentReference[oaicite:7]{index=7}  
-  Machine Learning libraries (e.g., scikit-learn, pandas, etc.) – for training and inference (assumed typical stack)

---

##  Getting Started

### Clone the Repository

```bash
git clone https://github.com/MasterPNJ/new-repository-game-predictor.git
cd new-repository-game-predictor
