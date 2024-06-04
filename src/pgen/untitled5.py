#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:02:49 2024

@author: matin
"""
import matplotlib.pyplot as plt
import pandas as pd

# Let's assume some example values for the weights (in kg) and MTOW (Maximum Takeoff Weight)
OE_weight = 24272 # Operational Empty Weight
payload_weight = 11500  # Payload Weight
MTOW = 43090
fuel_weight = MTOW-payload_weight
# Calculating the percentage of each weight with respect to MTOW
OE_weight_percentage = (OE_weight / MTOW) * 100
fuel_weight_percentage = (fuel_weight / MTOW) * 100
payload_weight_percentage = (payload_weight / MTOW) * 100

# Create a dataframe for the table
data = {
    'Weight Type': ['Operational Empty Weight (OE)', 'Fuel Weight', 'Payload Weight', 'Maximum Takeoff Weight (MTOW)'],
    'Absolute Value (kg)': [OE_weight, fuel_weight, payload_weight, MTOW],
    'Percentage of MTOW (%)': [OE_weight_percentage, fuel_weight_percentage, payload_weight_percentage, 100]
}

df = pd.DataFrame(data)

# Generating the pie chart
plt.figure(figsize=(8, 8))
plt.pie([OE_weight, fuel_weight, payload_weight], labels=['OE Weight', 'Fuel Weight', 'Payload Weight'],
        autopct='%1.1f%%', startangle=140)
plt.title('Weight Contributions for F100 Aircraft')
plt.show()

df

