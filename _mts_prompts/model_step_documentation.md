# Model Step Documentation

**Objective:** Provide detailed documentation for individual model steps using the MTS_MODEL_STEP_X keywords. This prompt dynamically identifies and documents each step in the model workflow, providing comprehensive technical specifications for each processing stage.

**Role:** As a Model Development Owner (MDO), you are responsible for documenting the detailed specifications and functionality of individual model steps in the Model Technical Specification (MTS) document.

**Task:**
• Identify all individual model steps using MTS_MODEL_STEP_X keyword patterns.
• Document the purpose, inputs, processing logic, and outputs for each step.
• Describe the technical implementation and algorithms used in each step.

**Instructions:**
• Scan the codebase for MTS_MODEL_STEP_X keywords to identify all model steps dynamically.
• For each identified step, provide:
  – Step purpose and objective
  – Input requirements and data dependencies
  – Processing logic and algorithms
  – Output specifications and format
  – Integration with other model steps
• Reference the relevant code files, functions, and classes associated with each step.
• Explain any mathematical formulations, statistical methods, or business logic applied in each step.
• Describe error handling and validation procedures for each step.
• Document any configuration parameters or settings specific to each step.
• Use clear, technical language suitable for model documentation.
• Base documentation strictly on the provided code and documentation without adding external information.

**Keywords:**
• MTS_MODEL_STEP_X: Used to reference specific steps in the model workflow (where X represents the step number or identifier).

## Expected Output Format

The output should be in JSON format with the following structure requirements:

1. **model_steps**: A comprehensive array containing detailed documentation for each identified model step
2. **step_dependencies**: A section mapping the relationships and dependencies between different model steps
3. **overall_workflow**: A high-level summary of how all steps work together in the complete model workflow

**Important Note:** If you do not know the information for any field, leave the value of that key as an empty string ("").

```json
{
  "model_steps": [
    {
      "step_id": "string",
      "step_name": "string",
      "purpose": "string",
      "inputs": {
        "description": "string",
        "variable_names": ["string"]
      },
      "process": {
        "description": "string",
      },
      "formulas": [
        {
          "formula_name": "string",
          "mathematical_expression": "string",
          "purpose": "string"
        }
      ],
      "outputs": {
        "primary_outputs": [
          {
            "name": "string",
            "type": "string",
            "format": "string",
            "description": "string"
          }
        ],
        "intermediate_outputs": [
          {
            "name": "string",
            "type": "string",
            "format": "string",
            "description": "string"
          }
        ]
      },
      "code_references": {
        "files": ["string"],
        "functions": ["string"],
        "classes": ["string"]
      },
      "configuration": {
        "settings": [
          {
            "parameter": "string",
            "value": "string",
            "description": "string"
          }
        ]
      }
    }
  ],
  "step_dependencies": [
    {
      "step_id": "string",
      "depends_on": ["string"],
      "provides_to": ["string"],
      "dependency_type": "string",
      "description": "string"
    }
  ],
  "overall_workflow": {
    "description": "string",
    "execution_order": ["string"],
    "total_steps": "string",
    "workflow_type": "string"
  }
}
```

**Note:** This prompt will automatically adapt to document any number of model steps found in the codebase, making it suitable for models with varying complexity and step counts. The documentation will be generated based on the actual MTS_MODEL_STEP_X patterns discovered in the Python (.py), Markdown (.md), and other documentation files.
