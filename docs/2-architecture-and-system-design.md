## Solution Architecture

<p align="center">
  <img src="https://github.com/AlphaKhaw/eco-predict/assets/87654386/9d90d91b-86cc-4bbd-976f-4a34b3a27385" alt="Architecture-Diagram" width="100%">
</p>

The diagram above illustrates the high-level architecture of our solution, focusing on the inner workings of each components but not the overall system integrating all components. The architecture diagram can be broken down into three main layers: User Interface Layer, Model Pipeline and Data Pipeline.

**User Interface Layer**:
- **Browser Interaction Node**: The entry point for the user, represented as the user's browser interaction with the system.
- **Streamlit**:
  - **Streamlit UI Service**: A lightweight and rapid development node for creating web applications, selected for its ease of use and efficiency.
  - **Streamlit Cloud Hosting**: This node represents the cloud service where the Streamlit application is hosted, ensuring scalability and accessibility without the need for local installations.
- **API Middleware Node**: Acts as the intermediary handling requests between the front-end Streamlit service and the backend model pipeline.
  - **FastAPI Service**: A robust framework designed for speed and ease of creating APIs, complete with automatic documentation and validation.
  - **Docker Containerization**: Encapsulates the FastAPI service to guarantee consistent execution across diverse environments.
  - **REST API Protocol**: A set of operations defining the interaction with the API, ensuring stateless communication via HTTP requests.
- **Cloud Hosting Node**: Depicts the hosting of the containerized FastAPI service on an AWS EC2 instance.

**Model Pipeline**:
- **Base Model Blueprint Node**: The foundational structure upon which main model classes are developed.
- **Random Forest and XGBoost Models**: A dual-model node representing the combination of Random Forest and XGBoost algorithms.
- **Model Trainer Processor**: This component takes charge of training the model with the right set of hyperparameters.
- **Model Inference Engine**: A node dedicated to making real-time predictions using the trained model on new data inputs.

**Data Pipeline**:
- **Data Source Node (Data.gov.sg)**: The primary entry point for data, here represented by the government-curated datasets.
- **Selenium Data Fetching Service**: A service node that uses web automation to ensure the latest data is consistently fetched for the system.
- **Data Downloader Storage**: Represents the initial data storage after acquisition, before processing.
- **Data Preprocessor Transformer**: A component where raw data is cleaned and transformed into a usable format.
- **Data Splitter**: The final stage of the data pipeline that segments the data into training and testing sets, ready for the model training phase.


## Model Design and Considerations

In the context of our project, we have chosen to prioritise interpretability and transparency in our model selection. This section explains the rationale behind opting for a Glassbox models and the implications of our dataset constraints.

**Consideration 1: Emphasis on Explainability**

- **Glassbox Models** are a type of machine learning model characterized by their transparency and explainability. They are particularly useful in scenarios where understanding the model's decision-making process is crucial.
- For instance, in financial services or healthcare, stakeholders may require insights into the model's reasoning to trust and act upon its predictions.

**Consideration 2: Dataset Constraints**

- With the dataset at our disposal being limited, it would be imprudent to employ **Neural Networks**. These complex models demand substantial data to generalize effectively; otherwise, they are prone to overfitting. In contrast, classical Machine Learning algorithms are more likely to provide robust performance with smaller datasets.
- The limited size of our dataset can also result in the training of **Neural Networks** being erratic, often converging to suboptimal solutions that do not generalize well to unseen data.

Given these considerations, we have framed our problem as a regression task, which is conducive to the use of glassbox models such as linear regression, decision trees, or ensemble methods like gradient boosting machines.

## Pipeline Orchestration with Prefect

To manage and automate our pipelines, we utilise Prefect, an advanced workflow management system. Prefect allows us to orchestrate our data and model pipelines, ensuring that each step is executed in the correct order, handling dependencies, and providing mechanisms for error handling and retries.

## Technology Stack

Our technology stack has been carefully chosen to support our architecture and ensure seamless integration between components:

- **Data Science**: Python, Pandas, Scikit-learn for data manipulation and model training.
- **Configuration Management**: Hydra is integrated to manage configurations, allowing us to easily switch between different setups without changing the codebase, making our application flexible and scalable.
- **Orchestration**: Prefect for scheduling and monitoring pipelines.
- **Containerization**: Docker to encapsulate our environment and dependencies.
- **Version Control**: Git for code management and collaboration.
- **Continuous Integration/Deployment**: GitHub Actions to automate testing and deployment processes.
- **User Interface (UI) Component**: Empowering the front end, Streamlit offers an intuitive interface for users to interact with our application. It stands out for its ability to quickly turn data scripts into shareable web apps.

Each element of the stack plays a vital role in realising the efficient and reliable operation of our system.
