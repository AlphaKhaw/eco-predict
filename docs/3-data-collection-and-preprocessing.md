## Data Source

**Overview of [data.gov.sg](http://data.gov.sg/)**

Singapore’s commitment to building a smart nation is epitomized by the launch of [data.gov.sg](http://data.gov.sg/) in 2011. This platform serves as a comprehensive repository of data compiled by over 70 public agencies, and it stands as a testament to the nation’s dedication to transparency and collaboration. The portal is designed to foster a community where citizens and developers can freely access public data to co-create innovative and practical solutions that address a myriad of needs.

**Functionality and User Engagement**

[data.gov.sg](http://data.gov.sg/) is engineered to do more than just serve as a passive data store; it is an active instrument aimed at making government data not only available but also relevant and comprehensible to the general public. This mission is achieved through:

- The strategic use of data visualizations that transform raw data into engaging and informative graphical representations.
- Data-driven narratives that provide insight into societal trends and government policies through meticulous data analysis and visualization techniques.
- The development of user-friendly dashboards that present high-quality data in an easily digestible format, allowing for quick insights at a glance.
- The incorporation of shareable charts and tables, enhancing the interactivity and dissemination of datasets.
- The curation of articles that leverage data to elucidate trends and governmental initiatives.

**Publication of Building Energy Performance Data**

This dataset contains the building energy performance data collected through BCA’s Building Energy Submission System (BESS), under the legislation on Annual Mandatory Submission of Building Information and Energy Consumption Data for Section 22FJ ‘Powers to Obtain Information’ of Building Control Act.

*Here’s an overview of its key aspects:*

- **Origin**: The data originates from the Building and Construction Authority’s Building Energy Submission System (BESS), a tool designed for the systematic collection of building energy consumption data.
- **Scope**: This dataset encompasses a broad spectrum of buildings across Singapore, providing a comprehensive overview of the country’s building energy profiles.
- **Purpose**: The data supports the Annual Mandatory Submission of Building Information and Energy Consumption Data, a critical process for regulatory compliance and environmental monitoring.
- **Relevance**: It is an indispensable resource for discerning patterns in building energy use, serving as a cornerstone for the development of predictive models and strategies aimed at energy optimization.
- **Accessibility**: Emphasizing the government's commitment to transparency, the dataset is readily accessible through Singapore’s official data portal, enabling stakeholders to easily obtain and utilize the information.

*The following specific datasets are included:*

- Building Energy Performance for the years 2019
- Building Energy Performance for the years 2020
- Building Energy Performance for the Commercial Buildings

*Overall, the collection of datasets contain the following features:*
- Detailed building information including names, addresses, sizes, and types.
- Information on BCA Green Mark Awards, recognizing sustainable building design and performance.
- Comprehensive annual energy usage data for a variety of building categories.

This meticulous publication of data underscores Singapore’s strategic approach to energy management and environmental stewardship, providing valuable insights for policy-makers, researchers, and the public alike.

## Data Preprocessing

### 1. Data Parser

The `DataParser` class in the script is designed to automate the process of downloading datasets from a specified URL, typically from an online data repository like `Data.gov.sg`. The script is configured using a **Hydra** configuration YAML file that provides a structured way to supply configurations.

**Main Components**

1. **Initialization (`__init__` method)**
   - Instantiates the `DataParser` object, setting up necessary attributes from the configuration, such as file paths and expected files.
   - Checks for the presence of expected files in the specified directory and initializes the web driver if files are missing.

2. **File Fetching and Downloading (`fetch_and_download` method)**
   - Uses the provided URL and xpaths from the configuration to navigate to a webpage and download missing datasets.
   - Manages the web driver to interact with the website elements to trigger dataset downloads.

3. **Web Element Interaction (`_click_element` method)**
   - A helper function to click web elements specified by their xpath, which is used to navigate through the site and initiate downloads.

4. **Web Driver Initialization (`_initialise_webdriver` method)**
   - Sets up the Chrome Webdriver with specified options such as running in headless mode and setting the default download directory.

5. **File Movement (`_move_files_from_downloads` method)**
   - After files are downloaded to a default location, this method moves them to a specified output directory.

6. **Dataset Download Navigation (`_navigate_and_download_dataset` method)**
   - Handles the specific navigation and interaction required to download a single dataset.

7. **File Existence Check (`check_expected_files` static method)**
   - Checks if the expected files are present in the output folder and returns a list of missing files.

**Configuration**

```YAML
url: "https://beta.data.gov.sg/collections/22/view"
expected_files:
  - ListingofBuildingEnergyPerformanceData2019.xlsx
  - ListingofBuildingEnergyPerformanceData2020.csv
  - ListingofBuildingEnergyPerformanceDataforCommercialBuildings.csv
downloads_foldername: "Downloads"
output_folderpath: "${hydra:runtime.cwd}/data/raw"
download_file_button_xpath: '//*[@id="__next"]/div/main/div/div[1]/div/button'
download_button_xpath: '//*[@id="chakra-modal-:r3:"]/div[3]/button'
dataset_xpaths:
  'ListingofBuildingEnergyPerformanceData2019.xlsx': '/html/body/div[3]/div/div[3]/div/section/div[2]/div/div[1]/div/div/div'
  'ListingofBuildingEnergyPerformanceData2020.csv': '/html/body/div[3]/div/div[3]/div/section/div[2]/div/div[2]/div/div/div'
  'ListingofBuildingEnergyPerformanceDataforCommercialBuildings.csv': '/html/body/div[3]/div/div[3]/div/section/div[2]/div/div[3]/div/div/div'
```
- `url`: The webpage URL from which the data files are to be downloaded.
- `expected_files`: A list of filenames that the script expects to find or download.
- `downloads_foldername`: The default folder where the browser is configured to download files.
- `output_folderpath`: The directory path where the downloaded files should be moved after being fetched.
- `download_file_button_xpath`: The xpath to the button on the web page that triggers the file download process.
- `download_button_xpath`: The xpath to the final download button that needs to be clicked to start the download.
- `dataset_xpaths`: A dictionary mapping filenames to their respective xpaths on the web page, used to locate the specific datasets for downloading.

**Workflow**

1. The script checks for the presence of expected files in the output directory.
2. If files are missing, the web driver is initialized and navigates to the specified URL.
3. For each missing file, the script uses the xpaths provided to locate and trigger the download of the dataset.
4. Once the files are downloaded to the default location, the script moves them to the specified output folder.

**Usage**

The script is designed to be used with a command-line interface where the Hydra framework manages the configuration. The user would execute the command below, and the `DataParser` class will automate the download and organization of the specified data files.

```bash
python -m src.data_preprocessing.data_parser
```

### 2. Data Preprocessor

**Main Component**

The `Data Preprocessor` is a key component designed to clean and prepare dataset(s) for analysis or model training. The preprocessing includes a series of steps defined in the configuration file, allowing customization for different datasets. The `preprocess_data` method orchestrates the entire preprocessing pipeline as outlined below:

1. **Configuration Reading**: The method starts by reading the configurations from `self.cfg`. These settings dictate the output folder paths, suffix for the processed files, and specific preprocessing instructions for each dataset.

2. **Data Reading**: Based on the dataset identifier or name provided, it reads the input file path from the config and loads the data into a dataframe.

3. **Column Operations**:
   - **Dropping Columns**: Removes unnecessary columns as specified in the config.
   - **Renaming Columns**: Updates column names to more descriptive or required formats.

4. **Data Cleaning**:
   - **Symbol Removal**: Cleans up specified symbols from the data, like commas or percentage signs.
   - **Data Type Conversion**: Converts columns to the required data types.

5. **Data Imputation**:
   - **Categorical Imputation**: Fills in missing values in categorical features.
   - **Numerical Imputation**: Applies strategies to handle missing numerical data.

6. **Data Encoding**: Transforms categorical data into a machine-readable format. This includes:
   - **Binary Encoding**: Encodes binary categorical features.
   - **Nominal Encoding**: Encodes nominal features without an intrinsic order.
   - **Ordinal Encoding**: Encodes ordinal features that have a defined hierarchy or order.

7. **Exporting**: The processed dataframe is then saved to the output folder with a name that combines the dataset identifier and the predefined suffix.


**Configuration**

- **General Settings**:
  - `output_folderpath`: Directory to save processed data.
  - `suffix`: Suffix for the output file names.
  - `common_identifier`: A common field used across datasets (not used in the script provided).

- **Dataset Specific Settings**: Each dataset has its own configuration, identified by a year or other identifier, with the following possible settings:
  - `enable_preprocessing`: Flag to enable/disable preprocessing for the dataset.
  - `input_filepath`: The location of the dataset file to process.
  - `preprocessing_steps`: Detailed steps for preprocessing, which include:
    - `drop_columns`: A list of column names to be dropped.
    - `remove_symbols`: A mapping of columns to symbols that should be removed.
    - `convert_dtypes`: A mapping of columns to the data types they should be converted to.
    - `impute_categorical_columns`: Instructions for imputing missing values in categorical columns.
    - `impute_numerical_columns`: Instructions for imputing missing values in numerical columns.
    - `encode_columns`: Instructions for encoding categorical columns which include sub-settings for binary, nominal, and ordinal encoding.
    - `rename_columns`: A mapping of old column names to new ones for renaming.

**Usage**

The script is designed to be used with a command-line interface where the Hydra framework manages the configuration. The user would execute the command below, and the `DataPreprocessor` class will automate the cleaning and transformation of the specified data files.

```bash
python -m src.data_preprocessing.data_preprocessor
```

### 3. Data Splitter

**Main Components**


1. **Initialization**:
   - Create a `DictConfig` object (using Hydra) with the required parameters.
   - Instantiate the `DataSplitter` class with this configuration.

2. **Splitting Data**:
   - Call the `split_data` method to perform the data splitting.
   - The method will read the configurations, check the validity of the input ratios, split the dataframe, and check the distribution of the split data.
   - Finally, it exports the train, validation, and test datasets to the specified output paths.

3. **Handling Splits**:
   - If `val_ratio` is set to `0.0`, the data is only split into train and test sets.
   - If `val_ratio` is non-zero, the data is split into train, validation, and test sets.

4. **Exporting Data**:
   - Split datasets are exported as separate files, named according to the `suffix` list in the configuration, to the `output_folderpath`.


**Configuration**

```YAML
general:
  input_filepath: "${data_processing.data_preprocessor.general.output_folderpath}/${data_processing.data_preprocessor.general.suffix}"
  output_folderpath: "${hydra:runtime.cwd}/data/split"
  suffix: ["train.csv", "test.csv"]
  seed: 42
ratios:
  train_ratio: 0.8
  val_ratio: 0.0
  test_ratio: 0.2
split_tolerance: 0.01
```
- `input_filepath`: Path to the input data file.
- `output_folderpath`: Path where the split data files will be saved.
- `suffix`: List of suffixes for the output files, corresponding to each data split.
- `seed`: Seed number for random operations to ensure reproducibility.
- `train_ratio`: Proportion of the data to be used for training.
- `val_ratio`: Proportion of the data to be used for validation.
- `test_ratio`: Proportion of the data to be used for testing.
- `split_tolerance`: The acceptable deviation between the expected and actual distribution of the data splits.

**Usage**

The script is designed to be used with a command-line interface where the Hydra framework manages the configuration. The user would execute the command below, and the `DataSplitter` class will automate the splitting of the specified data files for training and evaluation.

```bash
python -m src.data_preprocessing.data_splitter
```

## Data Pipeline Orchestration

**Role of `Prefect`**

The Data Pipeline using `Prefect` is designed to orchestrate a sequence of data operations in a robust and scalable manner. It ensures that each step of data handling, from parsing to preprocessing and finally splitting, is executed in a controlled environment. Prefect's framework provides monitoring and error-handling capabilities to manage workflows efficiently.

**Components**

## Prefect Data Pipeline Documentation

### Purpose

The Prefect Data Pipeline is designed to orchestrate a sequence of data operations in a robust and scalable manner. It ensures that each step of data handling, from parsing to preprocessing and finally splitting, is executed in a controlled environment. Prefect's framework provides monitoring and error-handling capabilities to manage workflows efficiently.

Prefect provides a dashboard to monitor the status of your pipelines, view logs, and get insights into the performance of different tasks.
If any task in the pipeline fails, Prefect has built-in retry and error-handling mechanisms to ensure robustness and continuity of your data workflows.

**Components**

The pipeline consists of three main tasks:

1. `parse_data`: This task is responsible for ingesting raw data and transforming it into a structured format. It typically involves reading data from various sources and parsing it into a format suitable for further processing.

2. `preprocess_data`: Once the data is parsed, this task performs a series of preprocessing steps. These steps may include cleaning, normalization, feature extraction, and any other transformations needed to prepare the data for modeling.

3. `split_data`: The final task in the pipeline is responsible for dividing the processed data into training, validation, and testing sets. This is crucial for training machine learning models, allowing for both model training and evaluation on separate data subsets to prevent overfitting.

**Usage**

1. Start Local Server:

    ```bash
    prefect server start
    ```

2. Checkout dashboard at: http://127.0.0.1:4200

3. Schedule a run for Data Pipeline flow:

    ```bash
    prefect deployment run 'Data Pipeline/test-run'
    ```

4. Use Python command to trigger run with logs:

    ```bash
    python -m src.pipelines.run_data_pipeline
    ```
