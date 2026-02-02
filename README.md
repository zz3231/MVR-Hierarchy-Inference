# MVR Hierarchy Inference Website

Interactive web application for demonstrating the Minimum Violation Ranking (MVR) algorithm.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Website

```bash
streamlit run mvr_website.py
```

The website will open automatically in your browser at `http://localhost:8501`

## Features

### Interactive Parameters
- **R (Repetitions)**: Adjust 100-5000 (default: 3000)
- **T (Iterations)**: Adjust 100-3000 (default: 1500)
- **Early Stopping**: Optional convergence detection
- **K-means Method**: Choose from 4 methods

### Visualizations
1. **Convergence Plot**: Track optimal ranking discovery over time
2. **Job Variance Analysis**: See which jobs have stable/uncertain positions
3. **K-means Curves**: View inertia/Q(K) curves for each method
4. **Cluster Visualization**: Interactive scatter plots showing job layers
5. **Detailed Assignments**: Expandable view of which jobs belong to which layer

### Four K-means Methods
1. **Bonhomme et al. (2019)** - Rank Std Threshold
2. **Overall Std Threshold** - Variance-based
3. **Elbow Method** - Manual K selection with visual curve
4. **Simple Variance** - Median variance threshold

## Test Data

- 10 Sales workers: Sales1 → Sales2 → ... → Sales6 → CEO
- 10 Engineering workers: Eng1 → Eng2 → Eng3 → Eng4 → CEO
- Theoretical maximum optimal rankings: C(10,6) = 210

## Project Structure

```
corp/
├── mvr_website.py              # Main Streamlit app
├── mvr_website_prototype.ipynb # Jupyter notebook prototype
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage Tips

1. **Start with default parameters** (R=3000, T=1500) to see full results
2. **Try different K-means methods** to compare hierarchy inference
3. **Use Elbow method** if you want to visually choose K (recommended K=4)
4. **Enable early stopping** for faster runs when R is large
5. **Lower R/T values** for quick testing (e.g., R=500, T=500)

## Deployment

### Deploy to Streamlit Cloud (Free)

1. Push this code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Select `mvr_website.py` as the main file
5. Deploy!

### Deploy to Other Platforms

- **Heroku**: Use `Procfile` with web command
- **Google Cloud Run**: Containerize with Docker
- **AWS EC2**: Run with nginx reverse proxy
- **Local Network**: Use `--server.address 0.0.0.0` flag

## Customization

To modify the test data, edit the `create_test_data()` function in `mvr_website.py`:

```python
def create_test_data():
    # Modify career paths here
    sales_path = ['Sales1', 'Sales2', 'Sales3', ...]
    eng_path = ['Eng1', 'Eng2', ...]
    ...
```

## Performance Notes

- **R=3000, T=1500**: ~30-60 seconds
- **R=1000, T=1000**: ~10-20 seconds
- **R=500, T=500**: ~5-10 seconds
- Progress bar shows real-time status

## Troubleshooting

**Website won't start?**
```bash
# Make sure Streamlit is installed
pip install streamlit --upgrade
```

**Plots not showing?**
```bash
# Reinstall matplotlib
pip install matplotlib --upgrade
```

**Too slow?**
- Lower R and T values
- Enable early stopping
- Use a faster machine

## Contact

For questions about the MVR algorithm, refer to:
- Paper: Huitfeldt et al. (2023) - "Internal labor markets: A worker flow approach"
- Notebook: `mvr_website_prototype.ipynb`
