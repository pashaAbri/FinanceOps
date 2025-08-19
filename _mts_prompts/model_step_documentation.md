# Model Step Documentation

**Objective:** Provide detailed documentation for a specific individual model step using the MTS_MODEL_STEP_X keyword. This prompt focuses on documenting one designated step in the model workflow, providing comprehensive technical specifications for that specific processing stage.

**Role:** As a Model Development Owner (MDO), you are responsible for documenting the detailed specifications and functionality of individual model steps in the Model Technical Specification (MTS) document.

**Task:**
• Document the specified individual model step using the provided MTS_MODEL_STEP_X keyword.
• Provide comprehensive documentation for the purpose, inputs, processing logic, and outputs of the specified step.
• Describe the technical implementation and algorithms used in the specified step.

**Instructions:**
• Focus on the specified model step identified by the MTS_MODEL_STEP_X keyword provided as input.
• For the specified step, provide:
  – Step purpose and objective
  – Input requirements and data dependencies
  – Processing logic and algorithms
  – Output specifications and format
  – Integration with other model steps
• Reference the relevant code files, functions, and classes associated with the specified step.
• Explain any mathematical formulations, statistical methods, or business logic applied in the specified step.
• Describe error handling and validation procedures for the specified step.
• Document any configuration parameters or settings specific to the specified step.
• Use clear, technical language suitable for model documentation.
• Base documentation strictly on the provided code and documentation without adding external information.

**Input Parameters:**
• **STEP_TO_DOCUMENT**: The specific MTS_MODEL_STEP_X keyword to document (e.g., "MTS_MODEL_STEP_1", "MTS_MODEL_STEP_2", etc.)

**Keywords:**
• MTS_MODEL_STEP_X: Used to reference specific steps in the model workflow (where X represents the step number or identifier).

## Expected Output Format

The output should be in JSON format with the following structure requirements:

1. **model_step**: Detailed documentation for the specified model step
2. **step_dependencies**: Dependencies and relationships of this specific step with other steps in the workflow
3. **integration_context**: How this step integrates within the overall model workflow

**Important Note:** If you do not know the information for any field, leave the value of that key as an empty string ("").

```json
{
  "model_step": {
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
  },
  "step_dependencies": {
    "depends_on": ["string"],
    "provides_to": ["string"],
    "dependency_type": "string",
    "description": "string"
  },
  "integration_context": {
    "position_in_workflow": "string",
    "workflow_phase": "string",
    "critical_path": "string",
    "execution_sequence": "string"
  }
}
```

**Note:** This prompt focuses on documenting a single specified model step, allowing for detailed analysis of individual workflow components. Each call to this prompt should specify one MTS_MODEL_STEP_X keyword to document. For complete model documentation, multiple calls should be made, one for each step in the workflow. The documentation will be generated based on the specified MTS_MODEL_STEP_X pattern found in the Python (.py), Markdown (.md), and other documentation files.
