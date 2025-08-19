# Data Relationship

Given the following model_relationships:

```
[add the contents of model_relationships.md]
```

## Expected Output Format

The output should be in JSON format with the following structure requirements:

1. **upstream**: Models that feed into this model, including their names and explanations of how they are used
2. **downstream**: Models that consume outputs from this model, including their names and explanations of how they use the outputs

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
