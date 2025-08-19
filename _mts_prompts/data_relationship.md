# Data Relationship

**Objective:** Using the model workflow documentation, relationship mappings, and parameter statistics, describe the relationships between data elements in the model. Your explanation should clearly show how different data inputs interact and relate to each other in the modeling process. For references, look for MTS_DATA_RELATIONSHIP keyword.

**Role:** As the Model Development Owner (MDO), you are responsible for documenting the relationships and interactions between data elements for the model in the Model Technical Specification (MTS) document.

**Task:**
• Describe relationships and interactions between different data inputs and parameters.
• Map data dependencies and flow between model components.
• Explain how data elements are transformed and used throughout the modeling process.

**Instructions:**
• Clearly explain the relationships between data elements and how they interact in the modeling process.
• Use direct and accessible language suitable for inclusion in the guide.
• Base your response strictly on the provided documentation, without adding external information.
• Use other relevant documentation as references.
• Knowledge base keywords are included in .py and .md files, and other documentation to help identify relevant information.

## Expected Output Format

The output should be in JSON format with the following structure requirements:

1. **upstream**: Models or data sources that feed into this model, including their names and explanations of how they are used
2. **downstream**: Models or processes that consume outputs from this model, including their names and explanations of how they use the outputs

**Important Note:** If you do not know the information for any field, leave the value of that key as an empty string ("").

```json
{
  "upstream": [
    {
      "name": "string",
      "explanation": "string"
    }
  ],
  "downstream": [
    {
      "name": "string",
      "explanation": "string"
    }
  ]
}
```
