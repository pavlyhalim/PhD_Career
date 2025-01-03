# PhD Career Advisor Application

A Streamlit application and a machine learning model to predict career personas for PhD alumni and provide tailored career recommendations.

## Project Overview

- **cluster_model.py**: Contains the `PersonaModel` class, which handles data preprocessing, clustering, and classification for predicting career personas.
- **app.py**: Streamlit application that uses the `PersonaModel` to generate career recommendations based on user input.

## Installation

1. **Set Up Environment**

   Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install Dependencies**

   Install the required packages:

   ```bash
   pip install -r requirements.txt  # Ensure a requirements.txt file is present
   ```

3. **Obtain API Keys**

   - **Google API Key**: For Google Generative AI.
   - **SerpAPI Key**: For search functionalities.

   Store these keys in a `secrets.toml` file in the root directory:

   ```toml
   [secrets]
   GOOGLE_API_KEY = "your-google-api-key"
   SERPAPI_KEY = "your-serpapi-key"
   ```

## Usage

1. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

2. **Access the Application**

   - Open a web browser and navigate to `http://localhost:8501/`.
   - Follow the instructions in the app to input your profile and receive career recommendations.

## Adding Features

### To the Streamlit App

1. **Modify `app.py`**

   - **New Tabs**: Add new tabs using `st.tabs()` for additional sections.
   - **New Functionality**: Implement new functions or modify existing ones to include additional features.
   - **UI Elements**: Use Streamlit's widgets (e.g., `st.slider`, `st.selectbox`) to gather more user input.

2. **Update the Agent**

   - **New Tools**: Add new tools to the agent in `create_agent_with_tools()`.
   - **Extended Prompts**: Modify prompts to incorporate new data or recommendations.

### To the Machine Learning Model

1. **Update `PersonaModel`**

   - **New Features**: Add new features to `clustering_features`, `predictor_vars`, or other attributes.
   - **Model Tuning**: Experiment with different clustering or classification algorithms.

2. **Retrain the Model**

   - **Data Preprocessing**: Ensure new data is preprocessed correctly.
   - **Fit the Model**: Update the `fit()` method to include new data or parameters.

## Removing Features

### From the Streamlit App

1. **Delete or Comment Out Code**

   - **Tabs**: Remove or comment out code related to the tab you wish to eliminate.
   - **Functions**: Delete functions or features no longer needed.
   - **UI Elements**: Remove widgets or sections from the user interface.

2. **Adjust Agent Tools**

   - **Remove Tools**: If a tool is no longer needed, remove it from the agent's toolset.

### From the Machine Learning Model

1. **Modify `PersonaModel`**

   - **Remove Features**: Exclude features from `clustering_features`, `predictor_vars`, etc.
   - **Update Methods**: Adjust preprocessing, clustering, or classification methods accordingly.

2. **Retrain the Model**

   - **Exclude Data**: Ensure that the model is trained without the removed features.

## Troubleshooting

- **API Key Errors**: Ensure that API keys are correctly placed in `secrets.toml`.
- **Model Loading Issues**: Verify that the model files exist in the specified directory.
- **Streamlit Errors**: Check the terminal for error messages and correct them accordingly.

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://langchain.readthedocs.io/en/latest/)
- [Google Generative AI](https://developers.generativeai.google/)
- [SerpAPI](https://serpapi.com/)
