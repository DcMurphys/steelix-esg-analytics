# Time-series Forecasting Project: Environmental Aspect of Steel Plant
# Business Understanding 
## Company Profile
Steelix Plant Corporation stands as a robust leader in the global steel industry, renowned for its high-quality production of steel coils, steel plates, and iron plates. With a commitment to innovation and reliability, Steelix serves a diverse range of sectors, from construction and automotive to manufacturing and infrastructure development. Our state-of-the-art facilities are equipped with advanced machinery, enabling us to consistently deliver products that meet rigorous international standards.

As a company deeply ingrained in heavy industry, Steelix Plant Corporation's environmental footprint is acutely recognized. The challenges posed by increased CO2 emissions are acknowledged, particularly during the colder months at the beginning and end of the year. It's indicated by internal analyses that this seasonal surge in emissions often correlates with a significant increase in power usage (in kWh). This heightened energy demand is primarily driven by intensified active machine operation to maintain production targets, alongside the substantial energy required for water boiling processes crucial to winter operations.

## Problem to be Solved
At Steelix, these environmental considerations are being addressed head-on. Active investment is being made in research and development to explore sustainable practices, optimize energy efficiency, and implement cutting-edge technologies. The mitigation of seasonal CO2 emissions, while ensuring uninterrupted, high-quality production, is crucial. For these underlying problems to be effectively solved, the critical role of a data scientist is emphasized, as their expertise is needed to analyze complex energy consumption patterns, identify key drivers of emissions, and develop predictive models for sustainable solutions. The vision is not only for the demands of the modern world for essential steel products to be met, but for this to be achieved in a manner that champions environmental stewardship and contributes to a greener, more sustainable future.

**The aim of this project are as follows:**
* To accurately analyze complex energy consumption patterns (usage in kWh) throughout the year, with a particular focus on seasonal variations (beginning and end of the year).
* To identify and quantify the primary factors (e.g., nsm, day_of_week, lagging current, leading current) contributing to increased power usage and correlated CO2 emissions, especially during winter.
* To build and validate robust machine learning models capable of forecasting future energy consumption (kWh) and associated CO2 emissions, enabling proactive management.
* To inform and support initiatives aimed at optimizing energy efficiency across plant operations by providing data-driven insights and predictive capabilities.
* To provide data-driven recommendations and tools that contribute to the implementation of sustainable energy practices and cutting-edge technologies within the steel manufacturing process.

## Why These Problems Matters to Us?
The issue of carbon dioxide emissions and tremendous power consumption from industrial activities, as seen at Steelix Plant Corporation, holds immense significance for our collective environmental responsibility and sustainability. These emissions directly fuel global warming and climate change, leading to severe consequences like extreme weather and rising sea levels that impact everyone. Beyond this critical environmental toll, there are significant business imperatives. Stricter regulatory compliance is becoming the norm, with penalties for non-adherence and expectations to align with international agreements like the Paris Agreement. Companies now face strong pressure for corporate responsibility, where managing their environmental footprint enhances reputation and meets growing consumer and investor demands for sustainability. Furthermore, reducing emissions often translates to economic benefits through improved energy efficiency and cost savings, positioning companies as leaders in a green economy.* Ultimately, proactively addressing these emissions is vital for a company's long-term viability, ensuring it can thrive in a future where sustainability is no longer an option, but a core determinant of success.

**Reference taken from a paper article titled "The Rising Threat of Atmospheric CO2: A Review on the Causes, Impacts, and Mitigation Strategies" written by Nunes (2023).*

## Environment Preparation
In preparation for the project, a comprehensive set of libraries is imported to support various aspects of the data science workflow. The standard utility libraries such as os, warnings, and math are used for directory navigation, suppression of non-critical messages, and mathematical operations respectively. Additionally, the environment includes a directory scan to list available input files, which is particularly useful in managed environments like Kaggle.
For data manipulation and computation, the commonly used numpy and pandas libraries are included to handle numerical arrays and structured data. Time-based analysis is facilitated by the datetime and timedelta modules. Visualization is addressed with both static and interactive libraries—matplotlib.pyplot and seaborn for traditional plots, and plotly.graph_objects for dynamic and interactive visualizations.
In the preprocessing and modeling domain, the sklearn.preprocessing module is used to normalize and encode features, while model evaluation is handled using mean_squared_error. For building and training neural networks, the tensorflow.keras API is employed, utilizing layers such as Conv1D, MaxPooling1D, Flatten, and Dense, along with regularization and optimization techniques like Dropout and the Adam optimizer. Together, these libraries establish a robust foundation for data handling, visualization, preprocessing, and deep learning modeling.

## Data Preparation
The data preparation phase begins with loading the steel industry dataset from a CSV file using pandas. An initial assessment is performed to inspect the dataset’s structure, confirm the absence of missing values, and understand categorical distributions. A statistical overview of categorical columns is generated to support exploratory data analysis.

# Data Understanding
## Initial Exploration
Here are the brief information of these columns (attributes) related to the defined goals of this project:
* **date:** Timestamp indicating the exact moment of data capture, recorded typically every 15 minutes. This is crucial for tracking temporal patterns of energy use and emissions. 
* **usage_kWh:** Active power consumed by the plant during the recorded interval (in kilowatt-hours, kWh). This is the primary indicator of the plant's energy demand and its direct contribution to the environmental footprint. 
* **Lagging_Current_Reactive.Power_kVarh:** Reactive power consumption from inductive loads (e.g., motors, transformers) (in kilovolt-ampere reactive-hours, kVarh). High lagging reactive power indicates inefficient energy utilization, impacting overall sustainability. 
* **Leading_Current_Reactive_Power_kVarh:** Reactive power consumption from capacitive loads (in kilovolt-ampere reactive-hours, kVarh). Similar to lagging reactive power, it affects power quality and efficiency.
* **CO2(tCO2):** Concentration of carbon dioxide emissions observed at the plant (in parts per million, ppm). This is a direct measure of the plant's environmental impact from greenhouse gas emissions.
* **Lagging_Current_Power_Factor:** The power factor associated with inductive loads (dimensionless, typically between 0 and 1). A lower power factor indicates poorer electrical efficiency, leading to higher energy losses and increased CO2 emissions for the same output. 
* **Leading_Current_Power_Factor:** The power factor associated with capacitive loads (dimensionless, typically between 0 and 1). Similar to lagging, it's an indicator of electrical system efficiency.
* **NSM:** Number of seconds from midnight (0-86399). This cyclical temporal feature helps in understanding daily operational patterns and their influence on energy consumption and emissions.
* **WeekStatus:** Categorical indicator denoting whether the data sample was observed during a weekday or a weekend. This helps identify weekly variations in plant activity and energy demand.
* **Day_of_week:** Categorical value representing the specific day of the week (e.g., Monday, Tuesday, Sunday) when the sample was observed. Useful for discerning specific daily operational rhythms.
* **Load_Type:** Categorical classification of the power consumption level for each observed sample (e.g., 'light', 'medium', or 'maximum' load). This aids in characterizing the intensity of operations and its energy/emission implications.

The initial data exploration focuses on gaining familiarity with the dataset's structure, completeness, and content distribution. Upon loading the dataset, the number of rows and columns is examined to establish its dimensionality. A check for missing values is conducted to assess data quality and identify potential preprocessing needs. Special attention is given to categorical variables by isolating them and summarizing their frequency distributions. This provides insights into the nature and variety of categorical entries, which is essential for encoding decisions later in the pipeline. The process sets the stage for more advanced feature analysis and modeling by ensuring a foundational understanding of the dataset's key components.

## Feature Exploration
During feature exploration, the focus shifts to analyzing the behavior and distribution of numerical attributes in the dataset. Using customized visualization functions, such as box plots and line charts, the data is examined across different dimensions to identify patterns, trends, and outliers. These visual tools help in understanding how features like energy consumption (Usage_kWh) and carbon emissions (CO2(tCO2)) vary over time and across operational contexts. By investigating these relationships visually, the analysis highlights potential feature importance, variance, and anomalies, guiding the next steps in feature engineering and model development.

The exploratory data analysis conducted through visualizations reveals several key insights into the dataset's structure and behavior. 

**Question 1: What were the trends in energy and power factors during 2018?**
* Usage_kWh showed a decline in mid-year followed by a rise toward the end.
* Lagging_Current_Power_Factor declined until September, then sharply increased.
* Leading_Current_Power_Factor remained stable but dipped significantly in November.
* Reactive power values exhibited seasonal fluctuations, with noticeable peaks in May and November.
  
**Question 2: What was the trend of CO2 emissions throughout 2018?**
* CO2(tCO2) levels were highest at the beginning and end of the year, coinciding with winter months.
* Levels stabilized at a lower concentration from May to November, showing a clear seasonal pattern.

**Question 3: How was energy usage distributed across weekdays, week status, and load types?**
* Higher consumption occurred on weekdays compared to weekends.
* Medium loads were the most common, but maximum loads corresponded to the highest usage levels.
* Weekends and Sundays, in particular, showed significantly lower consumption and more outliers.

**Question 4: How does energy usage correlate with time of day (NSM)?**
* Energy usage followed a strong diurnal pattern, rising in the morning, peaking during working hours, and dropping in the evening.
* A sharp decline around noon indicated a lunch break, reinforcing the operational link between time and demand.

**Question 5: What are the correlations among numerical features?**
* Usage_kWh and CO2(tCO2) shared a near-perfect correlation (≈0.99), reflecting the environmental impact of energy use.
* Lagging_Current_Reactive.Power_kVarh also had strong positive correlations with both energy usage and emissions, suggesting its role in influencing operational loads.

# Data Preparation/Preprocessing
## Feature Engineering
The exploratory data analysis conducted through visualizations reveals several key insights into the dataset's structure and behavior. Box plots highlight the presence of variability and potential outliers in features such as energy usage (Usage_kWh) and reactive power factors, suggesting operational fluctuations across time. Time-series line plots illustrate clear temporal patterns, with both energy consumption and carbon emissions (CO2(tCO2)) showing cyclic trends, possibly aligned with daily or weekly industrial activity. These plots also uncover consistency in patterns for weekdays versus weekends, aligning with the WeekStatus feature. Together, the visualizations not only validate the relevance of temporal features but also provide a strong rationale for incorporating lagged variables and moving averages in subsequent modeling. This visual insight shapes the direction for feature engineering and model strategy in the forecasting phase.

## Data Transformation
Data transformation steps are applied to refine the dataset for modeling readiness. A key transformation involves resampling the original timestamped data to an hourly frequency, standardizing temporal intervals for consistent time-series analysis. This is accompanied by custom aggregation rules tailored to each feature—summing energy usage while averaging power factors and emissions. Furthermore, missing values generated during feature engineering are handled using targeted imputation strategies, such as filling lags and moving averages with domain-reasonable defaults. These transformations ensure that the dataset is temporally aligned, statistically coherent, and free of irregularities, establishing a robust foundation for subsequent machine learning modeling.

## Train/Test Split
The dataset is partitioned into training and testing subsets to enable model evaluation and prevent data leakage. The split respects the chronological order of the time-series data, ensuring that historical observations are used for training while future periods are reserved for testing. This approach simulates real-world forecasting scenarios and preserves temporal integrity. By maintaining this sequential split, the model is trained on past behavior and then assessed on its ability to generalize to unseen future data, laying the groundwork for reliable and interpretable performance evaluation.

## Encoding & Scaling
Following the train-test split, the data undergoes encoding and scaling to ensure compatibility with machine learning algorithms. Categorical variables are transformed into numerical format using techniques such as label encoding, which assigns a unique integer to each category. This enables seamless integration with deep learning models that require numerical input. Simultaneously, numerical features are scaled using normalization techniques like MinMaxScaler to bring all values into a uniform range. This step is critical for models sensitive to feature magnitude, such as neural networks, as it promotes faster convergence during training and prevents features with larger scales from dominating the learning process.

# Data Modeling
## Modeling with Seasonal Naive Model
As a baseline forecasting approach, the Seasonal Naive model is utilized to predict both the testing set and the energy and emissions data for the following month. This model assumes that future values will repeat the same seasonal patterns observed in the past—specifically using values from the same hour or day in the previous seasonal cycle, such as 24 hours or one week earlier. In this project, the model is configured to account for the strong daily and weekly seasonality present in energy consumption (Usage_kWh) and carbon emissions (CO2(tCO2)), making it a fitting choice for an industrial time-series context. Since it relies solely on historical data patterns, the model requires no parameter tuning or training, making it computationally efficient and easy to implement. While simple, it serves as a critical benchmark to evaluate the added value of more sophisticated deep learning models. Moreover, its application to forecast the next month provides a straightforward, interpretable baseline for practical scenario planning and model performance comparison.

## Modeling with Convolutional (CNN) Model
To enhance predictive accuracy beyond baseline models, a Convolutional Neural Network (CNN) is employed for time-series forecasting of energy usage (Usage_kWh) and carbon emissions (CO2(tCO2)) on the testing set. The CNN architecture is designed to capture local temporal patterns by applying 1D convolutional filters across the input sequences, effectively learning spatial dependencies in the time dimension. The model is built using the Sequential API from TensorFlow’s Keras module and consists of several key layers: one or more Conv1D layers for extracting local feature maps, followed by MaxPooling1D layers to reduce dimensionality and enhance computational efficiency. These are succeeded by Flatten and Dense layers, which transition the learned patterns into a fully connected network for final prediction. To prevent overfitting, Dropout layers are also incorporated.

~~~
# Example of convolutional sequential model used 
sample_conv_model = Sequential([
    Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=POOL_SIZE),
    Flatten(),
    Dense(units=DENSE_UNITS, activation='relu'), 
    Dropout(DROPOUT_RATE), 
    Dense(units=DENSE_UNITS, activation='relu'),
    Dense(units=1)
])
~~~

### Input and Convolution Layers 
* **Filters:** FILTERS — number of convolutional filters used to extract localized patterns from the input sequence.
* **Kernel Size:** KERNEL_SIZE — defines the width of the convolution window (i.e., how many time steps are considered at a time).
* **Activation:** ReLU — applies non-linearity to help the model learn complex relationships.
* **Input Shape:** Determined by the time window (N_STEPS) and the number of features in the training data.

### Pooling Layers
* **Purpose:** Reduces the dimensionality of the output from the convolutional layer, helping to down-sample the feature maps and reduce computational complexity.
* **Pool Size:** POOL_SIZE — retains every second value in the sequence.

### Miscellaneous Layers
* **Flatten():** Converts the pooled feature maps into a single one-dimensional vector to serve as input to the dense layers.
* **Dense(units=DENSE_UNITS, activation='relu'):** First dense layer with specified neurons applies ReLU activation to learn abstract patterns from the flattened features.
* **Dropout(rate=DROPOUT_RATE):** Randomly drops specified rate (in percent) of the neurons during training to prevent overfitting and promote generalization.
* **Dense(units=DENSE_UNITS, activation='relu'):** Another dense layer to deepen the representation capacity of the model.
* **Dense(units=1):** provides the final scalar prediction (e.g., one forecast value per time step).

### Training Parameters
* **Optimizer:** Adam with a learning rate of 2e-5, chosen for its adaptive learning ability and efficiency.
* **Epochs:** 75 — the number of full passes over the training dataset.
* **Batch Size:** 128 — number of samples processed before the model updates its weights.
* **Verbosity:** 1 — training progress is displayed with detailed output per epoch.

This structure is used consistently for both the Usage_conv_model and CO2_conv_model, allowing the model to learn from historical sequences and generalize to future patterns in energy and emissions data. The layered configuration is carefully chosen to balance model complexity, generalization, and computational performance.

# Evaluation
This structure is used consistently for both the Usage_conv_model and CO2_conv_model, allowing the model to learn from historical sequences and generalize to future patterns in energy and emissions data. The layered configuration is carefully chosen to balance model complexity, generalization, and computational performance.

## Analysis of Visualization
The visualization of forecast results for both Usage_kWh and CO2(tCO2) reveals notable differences between the Seasonal Naive and Convolutional models. The Seasonal Naive model, relying solely on repeated historical patterns, produces forecasts that exhibit rigid periodicity and often miss subtle shifts or anomalies in the actual test data. While it performs reasonably well in capturing coarse seasonal trends, its static nature limits its responsiveness to irregular variations. In contrast, the Convolutional Neural Network (CNN) model demonstrates a much closer alignment with the true values in the testing set. For Usage_kWh, the CNN effectively tracks both broad trends and short-term fluctuations, showing smoother and more dynamic forecast lines that adapt to recent changes in the data. Similarly, for CO2(tCO2), the CNN captures emission trends more precisely, with its predictions maintaining consistent proximity to actual values throughout the test period. Overall, the visual comparison confirms that the CNN model significantly outperforms the baseline by offering enhanced flexibility, sharper temporal resolution, and improved fidelity in both energy consumption and emission forecasting.

## Analysis of Root Mean Squared Error (RMSE) Results  
The RMSE evaluation provides a quantitative assessment of forecasting accuracy for both Usage_kWh and CO2(tCO2) using the Seasonal Naive and Convolutional models. As expected, the Seasonal Naive model yields higher RMSE values for both targets, reflecting its limitations in capturing nuanced patterns and sudden changes in the time series. Its reliance on rigid seasonal repetition results in forecasts that deviate significantly from the actual data, particularly during non-repetitive or irregular intervals. In contrast, the Convolutional Neural Network model achieves substantially lower RMSE scores, demonstrating its superior ability to learn complex temporal relationships and generalize beyond simple seasonality. For Usage_kWh, the CNN delivers a tighter fit to observed energy consumption values, while for CO2(tCO2), it consistently reduces error margins across the test period. This RMSE comparison not only confirms the improved predictive power of the CNN architecture but also validates its effectiveness in real-world industrial forecasting scenarios where temporal variability is prominent.

# Conclusion From This Project
In the end, the predictive modeling phase successfully established a robust framework for forecasting critical operational and environmental indicators at Steelix Plant Corporation. Our evaluation, primarily utilizing Root Mean Squared Error (RMSE), clearly demonstrated that the Convolutional Neural Network (CNN) model offered a modest yet significant improvement over the traditional Seasonal Naive approach. For both usage in kwh and CO2 concentration predictions over the test period, the CNN consistently exhibited a lower RMSE, with a difference of approximately 0.24 for each variable. This marginal outperformance underscores the CNN's enhanced capability to discern and capture more complex, nuanced temporal patterns inherent in the steel plant's energy consumption and emission data, paving the way for more refined and accurate forecasting.

These validated machine learning models directly contribute to the project's core objectives by providing sophisticated predictive capabilities. By accurately forecasting future energy consumption and associated CO2 emissions, especially considering the identified seasonal variations and the influence of factors like time of day and operational activities, invaluable data-driven insights are generated. This predictive power is crucial for proactive energy management, enabling Steelix Plant Corporation to inform and support initiatives aimed at optimizing energy efficiency across all plant operations. Ultimately, the project delivers foundational tools and data-driven recommendations that contribute to the implementation of more sustainable energy practices and cutting-edge technologies within the steel manufacturing process, aligning with commitment of this company to environmental stewardship.
