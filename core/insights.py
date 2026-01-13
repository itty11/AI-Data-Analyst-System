def generate_insights(schema, summary):
    insights = []

    if schema["rows"] < 100:
        insights.append("Dataset is small; insights may be limited.")

    if len(schema["numeric_columns"]) > 5:
        insights.append("Dataset contains multiple numerical features suitable for regression or clustering.")

    high_missing = [
        col for col, count in schema["missing_values"].items() if count > 0
    ]

    if high_missing:
        insights.append(f"Columns with missing values detected: {', '.join(high_missing)}")

    return insights
