# MTS Synthesizer - Complete Document Generator

**Objective:** Orchestrate the use of modular MTS prompts to generate a comprehensive Model Technical Specification (MTS) document. This synthesizer coordinates multiple specialized prompts to create cohesive, well-structured MTS sections.

**Role:** As a Model Development Owner (MDO), you are responsible for synthesizing information from multiple sources and specialized prompts to create complete MTS sections that provide comprehensive model documentation.

**Task:**
• Synthesize results from specialized MTS prompts into complete document sections.
• Combine outputs from multiple prompt results into cohesive MTS sections.
• Ensure consistency and flow between different sections of the MTS document.

**Synthesis Workflow:**

**Section 3.1 - Introduction:**
1. Receive results from `model_workflow_summary.md` containing overall model approach
2. Receive results from `literature_review_summary.md` containing theoretical foundations
3. Synthesize both results into a comprehensive introduction covering:
   - Model objectives and scope
   - Theoretical background and industry context
   - Overall approach and methodology

**Section 3.2.3.3 - Input/Feature/Parameter File:**
1. Receive results from `data_input_summary.md` containing all model inputs identification
2. Receive results from `data_input_documentation.md` containing detailed technical specifications
3. Receive results from `data_relationship.md` containing data interactions analysis
4. Synthesize all results into a comprehensive section covering:
   - Complete list of model inputs with descriptions
   - Technical specifications and requirements
   - Data relationships and dependencies
   - Parameter file specifications and usage

**Section 3.4 - Functional Flow/Workflow Logic:**
1. Receive results from `model_step_documentation.md` containing individual model steps documentation - **CRITICAL: Use this for detailed documentation of EACH individual step in the model**
2. Receive results from `model_workflow_summary.md` containing overall workflow approach - **CRITICAL: Use this to understand how different steps work together and the overall process flow**
3. Synthesize both step-by-step documentation and workflow summary results into a comprehensive workflow section covering:
   - Sequential flow of model processes (using workflow summary for overall flow)
   - Detailed description of each individual model step (using step documentation for specifics)
   - Input-output mappings for each step
   - Integration points and data flow between steps
   - How individual steps connect to form the complete workflow

**Instructions:**
• Receive and analyze the results from the relevant specialized prompts for each section.
• Synthesize the results from multiple prompt outputs into cohesive, well-structured sections.
• Ensure consistency in terminology, formatting, and technical detail across sections.
• Maintain proper MTS document structure with appropriate headers and organization.
• Cross-reference information between sections to ensure coherence and avoid redundancy.
• Use clear, professional language suitable for technical documentation standards.
• Validate that each section provides complete coverage of its designated topic area.

**Output Format:**
• Generate complete MTS sections with proper headers and sub-sections.
• Include all necessary technical details while maintaining readability.
• Ensure smooth transitions between subsections within each major section.
• Provide comprehensive coverage without duplication across sections.

**Quality Assurance:**
• Verify that synthesized content addresses all requirements of the original section objectives.
• Ensure technical accuracy and completeness across all synthesized sections.
• Confirm that the final output meets professional documentation standards.
• Validate that all relevant keywords and documentation sources have been properly utilized.

**Note:** This synthesizer prompt receives results from the modular prompt system and integrates them to generate complete, professional MTS documentation. It ensures that specialized prompt results are properly synthesized into comprehensive sections that meet all technical specification requirements.
