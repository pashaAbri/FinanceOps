# Data Input Documentation

**Objective:** Using the model workflow documentation and input specifications, create comprehensive documentation for all data inputs used in the model. Your explanation should provide detailed technical documentation for each input parameter and data source. For references, look for MTS_DATA_INPUT_DOCUMENTATION keyword.

**Role:** As developer of the model, you are responsible for documenting the detailed specifications and technical requirements for all data inputs in the model for the Model Technical Specification (MTS) document.

**Task:**
• Document detailed specifications for all model inputs, including formats and requirements.
• List required data files, variables, and their technical specifications.

**Instructions:**
• Provide detailed technical documentation for each data input with specifications and requirements.
• Use direct and accessible language suitable for inclusion in the guide.
• Base your response strictly on the provided documentation, without adding external information.
• Use other relevant documentation as references.
• Knowledge base keywords are included in .py and .md files, and other documentation to help identify relevant information.

## Expected Output Format

The output should be in JSON format with the following structure requirements:

1. **data_inputs**: A comprehensive section containing the list of all data inputs with detailed descriptions, technical specifications, and usage information
2. **parameters**: A comprehensive section covering required parameters, optional parameters, and configuration settings

**Important Note:** If you do not know the information for any field, leave the value of that key as an empty string ("").

```json
{
  "data_inputs": [
    {
      "name": "string",
      "file_path": "string",
      "data_type": "string",
      "description": "string",
      "source": "string",
      "primary_purpose": "string",
      "model_integration": "string",
      "preprocessing_steps": [
        "string"
      ],
      "data_flow": "string"
    }
  ],
  "parameters": {
    "required_parameters": {
      "parameter_name": {
        "type": "string",
        "description": "string",
        "default_value": "string"
      }
    },
    "optional_parameters": {
      "parameter_name": {
        "type": "string",
        "description": "string",
        "default_value": "string"
      }
    },
    "configuration_settings": {
      "setting_name": {
        "type": "string",
        "description": "string",
        "allowed_values": ["string"]
      }
    }
  }
}
```
