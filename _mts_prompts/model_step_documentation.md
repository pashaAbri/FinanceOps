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

**Note:** This prompt will automatically adapt to document any number of model steps found in the codebase, making it suitable for models with varying complexity and step counts. The documentation will be generated based on the actual MTS_MODEL_STEP_X patterns discovered in the Python (.py), Markdown (.md), and other documentation files.
