from pyspark.sql.functions import col, exp
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("DiabetesDiagnosis").getOrCreate()

df = spark.read.csv("diabetes.csv", header=True, inferSchema=True)

df = df.drop("Pregnancies", "SkinThickness")

df = df.na.drop()

def iqr_outlier_treatment(dataframe, columns, factor=1.5):
    """
    Detects and treats outliers using IQR for multiple variables in a PySpark DataFrame.

    :param dataframe: The input PySpark DataFrame
    :param columns: A list of columns to apply IQR outlier treatment
    :param factor: The IQR factor to use for detecting outliers (default is 1.5)
    :return: The processed DataFrame with outliers treated
    """
    for column in columns:
        # Calculate Q1, Q3, and IQR
        quantiles = dataframe.approxQuantile(column, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        # Define the upper and lower bounds for outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Filter outliers and update the DataFrame
        dataframe = dataframe.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))

    return dataframe

df = iqr_outlier_treatment(df, ["BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction"], factor=1.5)


columns_to_replace = ["BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction"]
for col_name in columns_to_replace:
    mean_val = df.filter(df[col_name] != 0).groupBy().mean(col_name).first()[0]
    df = df.withColumn(col_name, when(df[col_name] == 0, lit(mean_val)).otherwise(df[col_name]))
    
df = df.withColumn("BloodPressure/Insulin", (col("BloodPressure") / col("Insulin")))
df = df.withColumn("BloodPressure/BMI", (col("BloodPressure") / col("BMI")))
df = df.withColumn("Insulin/BMI", (col("Insulin") / col("BMI")))
df = df.withColumn("Insulin/Glucose", (col("Insulin") / col("Glucose")))
df = df.withColumn("BMI/Glucose", (col("BMI") / col("Glucose")))

# Select the useful columns for the vector assembler
useful_columns = ["Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "BloodPressure/Insulin", "BloodPressure/BMI"]

# Create the vector assembler
assembler = VectorAssembler(inputCols=useful_columns, outputCol="features")

# Transform the dataframe to include the vector column
df = assembler.transform(df)

# Split the data between test and train
train, test = df.randomSplit([0.7, 0.3], seed=42)

# Train Logistic Regression model
lr = LogisticRegression(labelCol="Outcome", featuresCol="features")
lr_model = lr.fit(train)

# Train Decision Tree model
dt = DecisionTreeClassifier(labelCol="Outcome", featuresCol="features")
dt_model = dt.fit(train)

# Train Random Forest model
rf = RandomForestClassifier(labelCol="Outcome", featuresCol="features")
rf_model = rf.fit(train)

# Evaluate models
evaluator = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction")

# Calculate metrics for Logistic Regression
lr_accuracy = evaluator.evaluate(lr_model.transform(test), {evaluator.metricName: "accuracy"})
lr_precision = evaluator.evaluate(lr_model.transform(test), {evaluator.metricName: "weightedPrecision"})
lr_recall = evaluator.evaluate(lr_model.transform(test), {evaluator.metricName: "weightedRecall"})

# Calculate metrics for Decision Tree
dt_accuracy = evaluator.evaluate(dt_model.transform(test), {evaluator.metricName: "accuracy"})
dt_precision = evaluator.evaluate(dt_model.transform(test), {evaluator.metricName: "weightedPrecision"})
dt_recall = evaluator.evaluate(dt_model.transform(test), {evaluator.metricName: "weightedRecall"})

# Calculate metrics for Random Forest
rf_accuracy = evaluator.evaluate(rf_model.transform(test), {evaluator.metricName: "accuracy"})
rf_precision = evaluator.evaluate(rf_model.transform(test), {evaluator.metricName: "weightedPrecision"})
rf_recall = evaluator.evaluate(rf_model.transform(test), {evaluator.metricName: "weightedRecall"})

# Print model performances
print(f"""
============================== Logistic Regression ============================== 
accuracy = {lr_accuracy * 100:.2f}%
precision = {lr_precision * 100:.2f}%
recall = {lr_recall * 100:.2f}% 
=================================================================================
""")

print(f"""
================================= Decision Tree =================================
accuracy = {dt_accuracy * 100:.2f}%
precision = {dt_precision * 100:.2f}%
recall = {dt_recall * 100:.2f}%
=================================================================================
""")

print(f"""
================================ Random Forest ==================================
accuracy = {rf_accuracy * 100:.2f}%
precision = {rf_precision * 100:.2f}%
recall = {rf_recall * 100:.2f}%
=================================================================================
""")




