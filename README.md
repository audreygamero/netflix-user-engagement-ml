# Netflix User Engagement & Recommendation System
This project explores how machine learning can improve user engagement and retention on a streaming platform like **Netflix**. We build a system that:
- Recommends movies based on content and platform availability 
- Predicts user churn with simulated behavioral data 
- Visualizes how recommendations increase engagement 
### Content-Based Movie Recommendation
- Combines genre, description, and streaming platform data
- Recommends movies using **TF-IDF** and **cosine similarity**
- Two types of recommendations:
  - Based on movie title
  - Based on favorite genres
### Churn Prediction Model
- Simulates 1,000 users with features like:
  - Average watch time
  - Genres watched
  - Months inactive
  - Premium status
- Trains a **Random Forest Classifier** to predict churn
- Outputs a **classification report** and **confusion matrix** (saved to `visuals/`)
### Engagement Analysis
- Simulates user watch time before and after personalized recommendations
- Shows improvement with a histogram
- Plot saved to: `visuals/engagement_comparison.png`
## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/netflix-user-engagement-ml.git
cd netflix-user-engagement-ml
```

2. Install the required libraries:
```bash
pip install -r requirements.txt
```
## Running the Code

- Run the recommender system:
```bash
python src/recommender.py
```
- Run churn prediction and save confusion matrix:
```bash
python src/churn_prediction.py
```
- Run the engagement visualization:
```bash
python src/engagement_plot.py
```

## Notes
- This is a simulation project — churn data is generated artificially.
- For any visualizations or charts used in the presentation, see the `visuals/` folder.

---
