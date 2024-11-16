---
author: "Daniela Rios"
output:
  html_document:
    mathjax: true
    keep_md: true
    highlight: zenburn
    theme:  spacelab
  pdf_document:
always_allow_html: true
---




![](stroke_1.jpg){width=900 height=200px}

****

<b><span style='color:#E888BB; font-size: 16px;'> 1 |</span> <span style='color:#000;'>Dataset Overview</span> </b>

<p style='color:#000;'>This dataset is intended to predict whether a patient is likely to suffer from a stroke based on several input parameters. These parameters include demographic information (such as age, gender, and marital status), medical history (including hypertension, heart disease), lifestyle factors (smoking status, work type), and physiological measurements (BMI, average glucose level).</p>

<p style='color:#000;'>Each row in the dataset represents a single patient and provides a snapshot of relevant details that can help predict the likelihood of experiencing a stroke. The goal is to use these features to train machine learning models that can identify patterns and make predictions about stroke occurrence.</p>

<p style='color:#000;'>Below is a detailed description of the available variables in the dataset, outlining each attribute and its corresponding meaning:</p>

<div style="color:white;display:fill;border-radius:8px;font-size:100%; letter-spacing:1.0px;">
  <p style="padding: 5px;color:white;text-align:center;">
    <b><span style='color:#E888BB'>Dataset Attribute Information</span></b>
  </p>
</div>

<style>
table {
  width: 100%; /* Ajuste del ancho de la tabla */
  max-width: 1200px; /* Máximo ancho */
  border-collapse: collapse;
  margin-left: auto; 
  margin-right: auto;
}

th, td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left; /* Justificar texto a la izquierda */
}

th {
  background-color: #fae0e4;
  color: black;
  text-align: center;
}

tr:nth-child(even) {
  background-color: #f2f2f2;
}
</style>

| Attribute             | Description                                                                                       |
|-----------------------|---------------------------------------------------------------------------------------------------|
| **id**                | Unique identifier for each patient                                                                |
| **gender**            | "Male", "Female" or "Other"                                                                        |
| **age**               | Age of the patient in years                                                                        |
| **hypertension**      | 0 if the patient doesn't have hypertension, 1 if the patient has hypertension                      |
| **heart_disease**     | 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease             |
| **ever_married**      | "No" or "Yes", indicating marital status                                                           |
| **work_type**         | Type of employment: "children", "Govt_job", "Never_worked", "Private", "Self-employed"             |
| **Residence_type**    | "Rural" or "Urban", indicating the residence type                                                  |
| **avg_glucose_level** | Average glucose level in the blood                                                                 |
| **bmi**               | Body mass index (BMI)                                                                             |
| **smoking_status**    | Smoking status: "formerly smoked", "never smoked", "smokes", or "Unknown"                         |
| **stroke**            | 1 if the patient had a stroke, 0 if not (target variable)                                          |

<p style='color:#000;'>The dataset was retrieved from Kaggle and is intended for educational and research purposes. The information is publicly available <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset" target="_blank" style="color:#0000EE;">here</a>.</p>

<b><span style='color:#E888BB; font-size: 16px;'> 2 |</span> <span style='color:#000;'>Exploratory Data Analysis (EDA)</span> </b>

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Introduction</strong></div>

This report presents an Exploratory Data Analysis (EDA) of a stroke dataset, containing various attributes of patients that may be related to their likelihood of experiencing a stroke. This analysis aims to provide insights into the key characteristics of the data and relationships among different variables, which will aid in understanding factors contributing to strokes.

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Data Loading and Cleaning</strong></div>


```r
library(ggplot2)
library(dplyr)
library(knitr)
library(kableExtra)
library(plotly)
library(RColorBrewer)
library(gridExtra)
library(reshape2)
```



```r
stroke_data <- read.csv("stroke.csv")
```

The variables are labeled for better readability and categorical variables are mapped to human-readable levels.


```r
variable_labels <- list(
  "id" = "ID",
  "gender" = "Gender",
  "age" = "Age",
  "hypertension" = "Hypertension",
  "heart_disease" = "Heart Disease",
  "ever_married" = "Ever Married",
  "work_type" = "Work Type",
  "Residence_type" = "Residence Type",
  "avg_glucose_level" = "Average Glucose Level",
  "bmi" = "Body Mass Index",
  "smoking_status" = "Smoking Status",
  "stroke" = "Stroke"
)

# Data Cleaning: Convert columns to appropriate data types
stroke_data$id <- as.factor(stroke_data$id)
stroke_data$gender <- factor(stroke_data$gender, levels = c("Male", "Female", "Other"), labels = c("Male", "Female", "Other"))
stroke_data$ever_married <- factor(stroke_data$ever_married, levels = c("No", "Yes"), labels = c("No", "Yes"))
stroke_data$work_type <- factor(stroke_data$work_type, levels = c("children", "Govt_job", "Never_worked", "Private", "Self-employed"), labels = c("Children", "Government Job", "Never Worked", "Private", "Self-Employed"))
stroke_data$Residence_type <- factor(stroke_data$Residence_type, levels = c("Rural", "Urban"), labels = c("Rural", "Urban"))
stroke_data$smoking_status <- factor(stroke_data$smoking_status, levels = c("formerly smoked", "never smoked", "smokes", "Unknown"), labels = c("Formerly Smoked", "Never Smoked", "Smokes", "Unknown"))
stroke_data$stroke <- factor(stroke_data$stroke, levels = c(0, 1), labels = c("No", "Yes"))
stroke_data$hypertension <- factor(stroke_data$hypertension, levels = c(0, 1), labels = c("No", "Yes"))
stroke_data$heart_disease <- factor(stroke_data$heart_disease, levels = c(0, 1), labels = c("No", "Yes"))

# Ensure numerical columns are numeric
stroke_data$age <- as.numeric(stroke_data$age)
stroke_data$avg_glucose_level <- as.numeric(stroke_data$avg_glucose_level)
stroke_data$bmi <- as.numeric(stroke_data$bmi)
```

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Handling Missing Values</strong></div>

To handle missing values, missing `bmi` values are imputed using the median.


```r
stroke_data$bmi[is.na(stroke_data$bmi)] <- median(stroke_data$bmi, na.rm = TRUE)
```


**Numerical Variables**The univariate analysis aims to understand the distribution and key characteristics of each variable in isolation. This provides insights into central tendencies, spread, and variability for numerical variables, as well as frequency distributions for categorical variables.

The dataset contains numerical variables such as `age`, `avg_glucose_level`, and `bmi`. Below, the distribution of these variables is visualized using density plots.


```r
numerical_columns <- c("age", "avg_glucose_level", "bmi")

for (col in numerical_columns) {
  label <- variable_labels[[gsub("\\.", "_", col)]]
  print(
    ggplot(stroke_data, aes_string(x = col)) +
      geom_density(fill = "steelblue", alpha = 0.7) +
      theme_minimal() +
      labs(title = paste("Density Plot of", label), x = tools::toTitleCase(gsub("_", " ", label)), y = "Density")
  )
}
```

![](stroke_EDA_files/figure-html/density-plots-1.png)<!-- -->![](stroke_EDA_files/figure-html/density-plots-2.png)<!-- -->![](stroke_EDA_files/figure-html/density-plots-3.png)<!-- -->

**Summary Statistics**

Summary statistics provide insights into the central tendencies and spread of the numerical variables.


```r
numerical_summary <- stroke_data %>%
  summarise(
    Age_Min = min(age, na.rm = TRUE),
    Age_Median = median(age, na.rm = TRUE),
    Age_Mean = mean(age, na.rm = TRUE),
    Age_Max = max(age, na.rm = TRUE),
    Age_SD = sd(age, na.rm = TRUE),
    Glucose_Min = min(avg_glucose_level, na.rm = TRUE),
    Glucose_Median = median(avg_glucose_level, na.rm = TRUE),
    Glucose_Mean = mean(avg_glucose_level, na.rm = TRUE),
    Glucose_Max = max(avg_glucose_level, na.rm = TRUE),
    Glucose_SD = sd(avg_glucose_level, na.rm = TRUE),
    BMI_Min = min(bmi, na.rm = TRUE),
    BMI_Median = median(bmi, na.rm = TRUE),
    BMI_Mean = mean(bmi, na.rm = TRUE),
    BMI_Max = max(bmi, na.rm = TRUE),
    BMI_SD = sd(bmi, na.rm = TRUE)
  )

# Reshape the summary to have a more readable format
summary_table <- tibble::tibble(
  Metric = c("Min", "Median", "Mean", "Max", "Standard Deviation"),
  Age = c(numerical_summary$Age_Min, numerical_summary$Age_Median, numerical_summary$Age_Mean, numerical_summary$Age_Max, numerical_summary$Age_SD),
  Glucose = c(numerical_summary$Glucose_Min, numerical_summary$Glucose_Median, numerical_summary$Glucose_Mean, numerical_summary$Glucose_Max, numerical_summary$Glucose_SD),
  BMI = c(numerical_summary$BMI_Min, numerical_summary$BMI_Median, numerical_summary$BMI_Mean, numerical_summary$BMI_Max, numerical_summary$BMI_SD)
)

# Display the summary table
kable(summary_table, "html") %>%
  kable_styling(full_width = F, bootstrap_options = c("striped", "hover", "condensed"))
```

<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Metric </th>
   <th style="text-align:right;"> Age </th>
   <th style="text-align:right;"> Glucose </th>
   <th style="text-align:right;"> BMI </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Min </td>
   <td style="text-align:right;"> 0.08000 </td>
   <td style="text-align:right;"> 55.12000 </td>
   <td style="text-align:right;"> 10.300000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Median </td>
   <td style="text-align:right;"> 45.00000 </td>
   <td style="text-align:right;"> 91.88500 </td>
   <td style="text-align:right;"> 28.100000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Mean </td>
   <td style="text-align:right;"> 43.22661 </td>
   <td style="text-align:right;"> 106.14768 </td>
   <td style="text-align:right;"> 28.862035 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Max </td>
   <td style="text-align:right;"> 82.00000 </td>
   <td style="text-align:right;"> 271.74000 </td>
   <td style="text-align:right;"> 97.600000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Standard Deviation </td>
   <td style="text-align:right;"> 22.61265 </td>
   <td style="text-align:right;"> 45.28356 </td>
   <td style="text-align:right;"> 7.699562 </td>
  </tr>
</tbody>
</table>

**Categorical Variables**

The frequency distributions of categorical variables provide insights into the proportion of patients in different categories.


```r
categorical_columns <- c("gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status", "stroke")

for (col in categorical_columns) {
  label <- variable_labels[[gsub("\\.", "_", col)]]
  print(
    ggplot(stroke_data, aes_string(x = col, fill = col)) +
      geom_bar(alpha = 0.8) +
      geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5, size = 3) +
      theme_minimal() +
      labs(title = paste("Frequency Distribution of", label), x = tools::toTitleCase(gsub("_", " ", label)), y = "Count", fill = tools::toTitleCase(gsub("_", " ", label))) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )
}
```

![](stroke_EDA_files/figure-html/bar-plots-1.png)<!-- -->![](stroke_EDA_files/figure-html/bar-plots-2.png)<!-- -->![](stroke_EDA_files/figure-html/bar-plots-3.png)<!-- -->![](stroke_EDA_files/figure-html/bar-plots-4.png)<!-- -->![](stroke_EDA_files/figure-html/bar-plots-5.png)<!-- -->![](stroke_EDA_files/figure-html/bar-plots-6.png)<!-- -->![](stroke_EDA_files/figure-html/bar-plots-7.png)<!-- -->![](stroke_EDA_files/figure-html/bar-plots-8.png)<!-- -->

**Scatter Plots for Numerical Variables**The multivariate analysis focuses on examining relationships between multiple variables simultaneously, such as correlations and associations. This helps to identify patterns and interactions that may contribute to stroke occurrences.

The relationships between pairs of numerical variables are explored using scatter plots.


```r
scatter_pairs <- combn(numerical_columns, 2)

for (i in 1:ncol(scatter_pairs)) {
  var1 <- scatter_pairs[1, i]
  var2 <- scatter_pairs[2, i]
  label1 <- variable_labels[[gsub("\\.", "_", var1)]]
  label2 <- variable_labels[[gsub("\\.", "_", var2)]]
  print(
    ggplot(stroke_data, aes_string(x = var1, y = var2)) +
      geom_point(alpha = 0.6, color = "darkgray") +
      geom_smooth(method = "lm", se = FALSE, color = "blue") +
      labs(title = paste("Scatter Plot of", label1, "vs", label2), x = tools::toTitleCase(gsub("_", " ", label1)), y = tools::toTitleCase(gsub("_", " ", label2))) +
      theme_minimal()
  )
}
```

![](stroke_EDA_files/figure-html/scatter-plots-1.png)<!-- -->![](stroke_EDA_files/figure-html/scatter-plots-2.png)<!-- -->![](stroke_EDA_files/figure-html/scatter-plots-3.png)<!-- -->

**Box Plots to Compare Categorical and Numerical Variables**

To further explore the data, numerical variables are compared across different categories using box plots.


```r
for (num_col in numerical_columns) {
  num_label <- variable_labels[[gsub("\\.", "_", num_col)]]
  for (cat_col in categorical_columns) {
    cat_label <- variable_labels[[gsub("\\.", "_", cat_col)]]
    print(
      ggplot(stroke_data, aes_string(x = cat_col, y = num_col, fill = cat_col)) +
        geom_boxplot(alpha = 0.7) +
        theme_minimal() +
        labs(title = paste("Box Plot of", num_label, "by", cat_label), x = tools::toTitleCase(gsub("_", " ", cat_label)), y = tools::toTitleCase(gsub("_", " ", num_label)), fill = tools::toTitleCase(gsub("_", " ", cat_label))) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    )
  }
}
```

![](stroke_EDA_files/figure-html/box-plots-1.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-2.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-3.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-4.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-5.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-6.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-7.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-8.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-9.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-10.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-11.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-12.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-13.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-14.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-15.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-16.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-17.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-18.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-19.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-20.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-21.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-22.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-23.png)<!-- -->![](stroke_EDA_files/figure-html/box-plots-24.png)<!-- -->

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Correlation Analysis</strong></div>

A correlation matrix is computed to explore the linear relationships between numerical variables. Below, significant correlations are presented.


```r
num_data <- stroke_data %>% select(all_of(numerical_columns))

# Ensure that num_data only contains numeric columns
num_data <- num_data %>% mutate_if(is.factor, as.numeric)

cor_matrix <- cor(num_data, use = "complete.obs")

upper_triangle <- cor_matrix
upper_triangle[lower.tri(upper_triangle, diag = TRUE)] <- NA

cor_data <- melt(upper_triangle, na.rm = TRUE)
cor_data <- cor_data %>%
  filter(!is.na(value)) %>%
  arrange(desc(abs(value)))

colnames(cor_data) <- c("Variable 1", "Variable 2", "Correlation")

kable(cor_data, "html") %>%
  kable_styling(full_width = F, bootstrap_options = c("striped", "hover", "condensed"))
```

<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Variable 1 </th>
   <th style="text-align:left;"> Variable 2 </th>
   <th style="text-align:right;"> Correlation </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:left;"> bmi </td>
   <td style="text-align:right;"> 0.3242957 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:left;"> avg_glucose_level </td>
   <td style="text-align:right;"> 0.2381711 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> avg_glucose_level </td>
   <td style="text-align:left;"> bmi </td>
   <td style="text-align:right;"> 0.1668757 </td>
  </tr>
</tbody>
</table>

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Correlation Heatmap Visualization</strong></div>

To further illustrate the relationships between numerical variables in the dataset, a correlation heatmap is presented below. This visual representation highlights both the strength and direction of linear relationships between variables. Strong positive or negative correlations may provide valuable insights into the relationships that influence stroke occurrences.


```r
# Filter numerical data for correlation
num_data <- stroke_data %>% select(all_of(c("age", "avg_glucose_level", "bmi")))

# Compute the full correlation matrix
cor_matrix <- cor(num_data, use = "complete.obs")

# Get the upper triangle of the correlation matrix without the diagonal
cor_matrix[lower.tri(cor_matrix, diag = TRUE)] <- NA

# Convert the matrix to a long format for plotting
cor_data <- melt(cor_matrix, na.rm = TRUE)

# Apply human-readable labels to variables in correlation data
cor_data$Var1 <- factor(cor_data$Var1, levels = colnames(num_data), labels = unlist(variable_labels[colnames(num_data)]))
cor_data$Var2 <- factor(cor_data$Var2, levels = colnames(num_data), labels = unlist(variable_labels[colnames(num_data)]))

# Create heatmap plot for pairwise correlations with correlation values on tiles
ggplot(cor_data, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 4) +
  scale_fill_gradient2(low = "red", high = "blue", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Correlation") +
  theme_minimal() +
  labs(
    title = "Correlation Heatmap of Numerical Variables",
    x = "",
    y = ""
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
```

![](stroke_EDA_files/figure-html/correlation-heatmap-1.png)<!-- -->


<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Conclusion</strong></div>

This EDA provides a detailed overview of the stroke dataset, focusing on both the individual characteristics of each variable and the relationships between variables. The visualizations and analyses highlight key factors that may contribute to stroke incidence and will serve as the foundation for further analysis and predictive modeling.



<br><br><br><br>
