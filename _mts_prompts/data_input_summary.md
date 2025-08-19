# Data Input Summary

**Objective:** Using the model workflow documentation, data summaries, and parameter statistics, create a comprehensive summary of all data inputs for the model. Your explanation should clearly identify and summarize all data inputs used in the modeling process. For references, look for MTS_DATA_INPUT_SUMMARY keyword.

**Role:** As the Model Development Owner (MDO), you are responsible for summarizing the data inputs and their characteristics for the model in the Model Technical Specification (MTS) document.

**Task:**
• Identify and summarize all data inputs used in the model.
• Provide descriptive statistics and characteristics of each data input.
• Describe the source and format of each data input.

**Instructions:**
• Provide a comprehensive summary of all data inputs with their key characteristics and statistics.
• Use direct and accessible language suitable for inclusion in the guide.
• Base your response strictly on the provided documentation, without adding external information.
• Use other relevant documentation as references.
• Knowledge base keywords are included in .py and .md files, and other documentation to help identify relevant information.

## Expected Output Format

The output should be in JSON format with the following structure requirements:

1. **data_inputs_summary**: Statistical summary for all data inputs including mean, median, standard deviation, min, max, count/shape
2. **parameters**: Actual parameter values used in the model

**Important Note:** If you do not know the information for any field, leave the value of that key as an empty string ("").

```json
{
  "data_inputs_summary": [
    {
      "name": "string",
      "statistics": {
        "mean": "string",
        "median": "string",
        "std_dev": "string",
        "min": "string",
        "max": "string",
        "count": "string",
        "shape": "string"
      }
    }
  ],
  "parameters": {
    "parameter_name": "actual_value"
  }
}
```
