# MTS Synthesizer - Complete Document Generator

**Objective:** Using the established MTS template structure (MTS_TEMPLATE), synthesize the outputs produced by modular MTS prompts to generate a comprehensive Model Technical Specification (MTS) document. This synthesizer takes the results from multiple specialized prompt outputs and integrates them to fill out the template sections and create cohesive, well-structured MTS documentation.

**Role:** As a Synthesizer, you are responsible for taking the information produced by other specialized MTS prompts and synthesizing it into a complete Model Technical Specification (MTS) document. Your role is to integrate and organize the outputs from multiple prompt executions to produce comprehensive model documentation.

**Task:**
• Take the outputs produced by specialized MTS prompts and integrate them into the MTS template structure.
• Synthesize information from multiple prompt results to populate each section of the MTS document.
• Only fill out the sections that are covered by the other specialized prompts - leave template sections as placeholders if no corresponding prompt output exists.
• Ensure consistency, coherence, and professional formatting across all synthesized sections.
• Create a complete, well-structured MTS document that follows the established template format.

**MTS Document Sections to Generate:**

The following sections represent the complete MTS template structure that needs to be populated using the outputs from specialized prompts:

**Introduction:**
Synthesize the results from `literature_review_summary.md` and `model_workflow_summary.md` to create a comprehensive introduction covering the model's purpose, theoretical background, and overall approach.
• Output should be a single paragraph (200-300 words)

**Input/Feature and Parameter File:**
Synthesize the results from `data_input_documentation.md`, `data_input_summary.md`, and `data_relationship.md` to create a comprehensive section covering model inputs and parameter specifications.
• Create a table with list of data variables including summary statistics and usage through the model
• Create another table with list of parameters including their values
• Include a paragraph explaining the upstream and downstream models

**Functional Form / Processing Logic:**
Synthesize the results from `model_step_documentation.md` and `model_workflow_summary.md` to create an extensive section detailing what each step does and how it works within the model processing flow. Use step documentation for detailed step information and workflow summary to understand how the steps are tied together.
• Provide comprehensive explanation of each step's functionality and implementation
• Detail how each step contributes to the overall model workflow
• Ensure the explanation properly addresses the order of the steps and the processing logic flows coherently

**Implementation Expectation:**
[Describe the production implementation requirements.]

**Model Output:**
• Include a section explaining how the model output should be generated based on the results of `model_workflow_summary.md`.
• Keep the output description short, approximately 50-100 words

**Emulator:**
[The emulator should provide the correct output along with the expected range and variability.]

**Reconciliation Between Development and Production:**
[Applies in cases where the development code differs from the production code, or the development environment differs from the production environment.]

**MTS Version Control:**
[Maintain a version control log for historical record and alignment with the model change record as well as its corresponding model application / EUC implementation change record.]

## Expected Output Format

The synthesizer should produce a complete MTS document following the template structure:

```markdown
# Model Technical Specification (MTS)

## Introduction
[Single paragraph (200-300 words) synthesized from literature review and workflow summary]

## Input/Feature and Parameter File
[Paragraph explaining upstream and downstream models]

### Data Variables
| Variable Name | Summary Statistics (mean, median, std deviation, min, max, shape) | Usage Through Model |
|---------------|-------------------------------------------------------------------|---------------------|
| [Variable 1]  | [Mean: X, Median: Y, Std: Z, Min: A, Max: B, Shape: C]          | [Usage description] |
| [Variable 2]  | [Mean: X, Median: Y, Std: Z, Min: A, Max: B, Shape: C]          | [Usage description] |

### Parameters
| Parameter Name | Value | Description |
|----------------|-------|-------------|
| [Parameter 1]  | [Val] | [Desc]      |
| [Parameter 2]  | [Val] | [Desc]      |

## Functional Form / Processing Logic
[Extensive section detailing each step's functionality and implementation]
[Detail how each step contributes to the overall model workflow]

## Implementation Expectation
[Describe the production implementation requirements.]

## Model Output
[Short description (50-100 words) explaining how the model output is generated based on the model workflow, including what the model produces, acceptable range and expected variability if applicable.]

## Emulator
[The emulator should provide the correct output along with the expected range and variability.]

## Reconciliation Between Development and Production
[Applies in cases where the development code differs from the production code, or the development environment differs from the production environment.]

## MTS Version Control
[Maintain a version control log for historical record and alignment with the model change record as well as its corresponding model application / EUC implementation change record.]
```

