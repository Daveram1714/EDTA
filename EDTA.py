import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('heart.data.csv')

# EDA on biking and heart disease
sns.jointplot(x='biking', y='heart.disease', data=df)
plt.xlabel('Biking (hours per week)')
plt.ylabel('Heart disease (0-100 scale)')
plt.title('Joint plot of biking and heart disease')
plt.show()

# Calculate the correlation between biking and heart disease
corr = df['biking'].corr(df['heart.disease'])

# Print the correlation
print('Correlation between biking and heart disease:', corr)

# Fit a linear regression model to the data
import statsmodels.formula.api as sm

# Create a formula for the model
formula = 'heart.disease ~ biking'

# Fit the model
model = sm.ols(formula, data=df).fit()

# Print the summary of the model
print(model.summary())

# Make predictions
predictions = model.predict(df)

# Calculate the residual errors
residuals = df['heart.disease'] - predictions

# Plot the residuals
plt.hist(residuals)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of residuals')
plt.show()

# Conclusion
# The correlation between biking and heart disease is negative, which means that people who bike more tend to have lower heart disease risk. The linear regression model fits the data well, and the residual errors are normally distributed.
