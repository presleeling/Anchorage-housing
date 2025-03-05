# Anchorage Housing Price Predictor

## Overview
This is a machine learning-powered web application for predicting housing prices in Anchorage, Alaska. The application uses a Random Forest Regressor to estimate house prices based on key features.

## Features
- Interactive web interface using Streamlit
- Machine learning price prediction
- Considers multiple housing characteristics:
  - Square footage
  - Number of bedrooms
  - Number of bathrooms
  - Year built
  - Neighborhood

## Project Structure
```
anchorage-housing-predictor/
│
├── main.py                 # Main Streamlit application
├── data_preprocessing.py   # Data generation and preprocessing
├── model_training.py       # Machine learning model training
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Steps
1. Clone the repository
```bash
git clone https://github.com/yourusername/anchorage-housing-predictor.git
cd anchorage-housing-predictor
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run main.py
```

## Model Details
- **Algorithm**: Random Forest Regressor
- **Features**: 
  - Square Feet
  - Number of Bedrooms
  - Number of Bathrooms
  - Year Built
  - Neighborhood

## Data Generation
Since real Anchorage housing data is not directly available in this example, the application uses:
- Synthetic data generation
- Simulated pricing based on realistic housing market assumptions
- Random Forest Regressor for price prediction

## Limitations
- Uses synthetic data
- Simplified pricing model
- Requires real-world data for accurate predictions

## Future Improvements
1. Integrate real Anchorage housing market data
2. Add more feature inputs
3. Implement more sophisticated feature engineering
4. Explore additional machine learning algorithms

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/anchorage-housing-predictor](https://github.com/yourusername/anchorage-housing-predictor)
