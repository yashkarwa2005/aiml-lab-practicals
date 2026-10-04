"""
AIML Laboratory - Semester 5
Assignment 1: Association Analysis Between Dependent and Independent Variables

Aim:
    To identify and analyze the association between dependent and independent
    variables using Pearson correlation techniques and scatter plots in Python.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("ASSIGNMENT 1: Association Analysis Between Variables")
    print("=" * 70)

    # Sample dataset: Study Hours (Independent Variable) vs Marks (Dependent Variable)
    data = {
        "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
        "Marks": [35, 42, 48, 55, 61, 68, 75, 82]
    }

    df = pd.DataFrame(data)
    print("\nDataset:")
    print(df.to_string(index=False))

    # Compute Pearson correlation coefficient
    correlation = df["Study_Hours"].corr(df["Marks"])
    print(f"\nPearson Correlation Coefficient (r): {correlation:.4f}")

    # Interpret correlation
    if correlation > 0.7:
        relationship = "Strong Positive Linear Association"
    elif correlation > 0.3:
        relationship = "Moderate Positive Linear Association"
    elif correlation > -0.3:
        relationship = "No Linear Association"
    elif correlation > -0.7:
        relationship = "Moderate Negative Linear Association"
    else:
        relationship = "Strong Negative Linear Association"

    print(f"Interpretation: {relationship}")

    # Scatter plot with regression trendline
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["Study_Hours"],
        df["Marks"],
        color="#1f77b4",
        s=80,
        edgecolors="black",
        linewidth=1,
        label="Observed Data"
    )

    # Linear trendline fit
    slope, intercept = np.polyfit(df["Study_Hours"], df["Marks"], deg=1)
    trendline = slope * df["Study_Hours"] + intercept
    plt.plot(
        df["Study_Hours"],
        trendline,
        color="#e63946",
        linestyle="--",
        linewidth=2,
        label=f"Trendline (y = {slope:.2f}x + {intercept:.2f})"
    )

    plt.title("Study Hours vs. Marks (Association Analysis)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Study Hours (Independent Variable)", fontsize=11)
    plt.ylabel("Marks Obtained (Dependent Variable)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()

    # Save figure and display
    plt.savefig("assignment_01_output.png", dpi=300)
    print("\nPlot saved as 'assignment_01_output.png'. Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
