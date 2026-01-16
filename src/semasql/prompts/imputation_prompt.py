selection_imputation_row = '''
Determine whether `{row}` satisfies the condition `{UDF_name}`.

Classify `{row}` as:
- "1" if condition `{UDF_name}` is satisfied.
- "0" if condition `{UDF_name}` is not satisfied.

Return only a single character: "0" or "1", with no explanation.
'''


projection_imputation_row = '''
# Task
Extract the value specified in the target column name from the source data.

## Input
- **Target Column**: "{new_col_name}"
- **UDF Information**: "{UDF_name}"
- **Source Column(s)**: "{col}"
- **Current Row Data**: {row}

---
### Examples
| UDF Information | Valid Outputs |
|-------------|---------------|
| `EvaluateBodyContentAsTrueOrFalse_from_Body` | `TRUE` or `FALSE` |
| `CheckEligibility_AnswerWithYesOrNo` | `YES` or `NO` |
| `ExtractYearsOfExperience` | `5` or `3.5` or `10+` ...|
| `GetCountryCode_TwoLetterISO` | `US` or `CA` ...|
---
## Output Rules
Return ONLY the extracted value with no explanations or extra text.
Match the exact format specified by the UDF name.
'''

binary_comparison = '''Criterion: "{UDF_name}"

Response A: {row_A}
Response B: {row_B}

Based on the {col} field(s), evaluate which response better satisfies the criterion "{UDF_name}".

Provide a brief one-sentence reasoning, then conclude with your answer.
Return `True` if Response A is better, return `False` if Response B is better.

Format:
Reasoning: [One sentence explaining your choice]
Answer: [True or False]'''

nested_loop_join_imputation_row = '''
Determine whether row pairs satisfy the condition "{UDF_name}".

{col_l}: {col_l_val}
{col_r}: {col_r_val}

Classify the result as:
- "1" if condition "{UDF_name}" is satisfied.
- "0" if condition "{UDF_name}" is not satisfied.

Return only a single character: "0" or "1", with no explanation.
'''

projection_join_imputation = '''
Map the value "{row}" from the "{col_l}" column to the corresponding value in the "{col_r}" column.

The list of available values in "{col_r}" is as follows:
[{val_list_r}]

Please return only the mapped value, with no explanation. If not mapped value exist, return "FALSE".
'''


join_imputation = '''
Task: Given two lists of distinct values from columns '{col_l}' and '{col_r}', determine which pairs satisfy the '{UDF_name}' condition.

Inputs:
- Column Left: '{col_l}' with values {col_l_value_list}
- Column Right: '{col_r}' with values {col_r_value_list}

Instructions:
- Evaluate each possible pair (from values in Column Left and Column Right) based on the '{UDF_name}' condition.
- Identify all pairs where the combination of values from Column Left and Column Right meet this condition.

Output Format:
Return the results in the following JSON format, including only the pairs that satisfy the condition.
To enhance the result, include your thinking process within the JSON as brief as possible:

{{
  "thinking": "{{...}}", 
  "result": 
  [
    {{"{col_l}": "value 1", "{col_r}": "value 3",}},
    ...
    {{"{col_l}": "value 2", "{col_r}": "value 4",}}
  ]
}}


Return the ouput strictly in the given format without any extra word outside JSON structure.
'''

aggregation_imputation = '''
Task: Aggregate Data Using "{UDF_name}" Operation

Instructions:
1. You have a dataset with the following values for "{col}": [{val_list}].
2. Apply the "{UDF_name}" operation to these values to compute an aggregated result.
3. Ensure that the aggregation helps in answering the following question: "{Question}".
4. Provide a concise, meaningful text-based summary of the key insights derived from the aggregated data.

Output Format:
- Generate a JSON array containing a single object.
- This object should have the key "agg_result" mapped to the aggregated value.
- Example Output:
[
  {{"agg_result": "aggregated_value"}}
]

Guidelines:
- Return only the JSON data itself.
- Do not include any additional text, formatting, or explanations. Avoid outputting the word 'json'.
'''

aggregation_imputation_with_groupby = '''
Task: Group and Aggregate Data

Instructions:
1. You have a dataset containing values for "{col}", represented by [{val_list}].
2. Group this data using the keys specified in {groupby_cols}.
3. Apply the "{UDF_name}" aggregation operation on all non-grouped columns to summarize the data  and provide a concise, meaningful text-based result. 
4. Ensure that the aggregation helps in answering the following question: "{Question}".
5. Focus on highlighting key insights or patterns derived from the aggregated data.

Input:
- Columns for grouping: "[{groupby_cols_val_list}]"

Output Format:
- Generate a JSON array where each entry corresponds to a group defined by the groupby columns.
- Each JSON object should map each column in {groupby_cols} to its value and include the aggregation result under the key "{UDF_name}.
- Example Output:
[
  {{"{groupby_cols[0]}": "value1", ..., "{UDF_name}": "result1"}},
  {{"{groupby_cols[0]}": "value2", ..., "{UDF_name}": "result2"}},
  ...
]

Guidelines:
- Return only the JSON data itself.
- Do not include any additional text, formatting, or explanations beyond the JSON structure. Avoid outputting the word 'json'.
'''


UDF_inline = '''
# Task
Analyze the provided User-Defined Function (UDF) and determine if it can be replaced with an equivalent SQL expression to avoid row-by-row execution.
Ensure the SQL expression is correct for answering the question: "{question}".

## UDF Information
- **Function Purpose**: {udf_prompt}
- **UDF Definition**: {udf} 
- **Input Column(s)**: {input_cols}
- **Sampled Input Values**: {sampled_values}

## Output format:
Response with a JSON format, containing two attributes "result" and "explanation".
  {{
    "result": "<SQL expression or NOT_REPLACEABLE>",
    "explanation": "<brief explanation with knowledge source/reasoning>"
  }}
- **If replaceable**: Provide the equivalent SQL expression and explain the logic/knowledge source
- **If not replaceable**: Return exactly `"NOT_REPLACEABLE"` as the result value

## Requirements

### Can Be Replaced

A UDF **can be replaced** when it performs deterministic operations expressible in native SQL, where the same input always produces the same output without external dependencies.

#### 1. Comparisons and Calculations with Known Values/Metrics
Use SQL when the logic involves fixed thresholds, authoritative constants, or mathematical formulas:
- `BMICategory(height_m, weight_kg)` → `CASE WHEN weight_kg/POWER(height_m,2) >= 30 THEN 'Obese' WHEN weight_kg/POWER(height_m,2) >= 25 THEN 'Overweight' ELSE 'Normal' END`
- `IsLegalDrinkingAgeUS(age)` → `age >= 21`
- `IsBoilingWater(temp_celsius)` → `temp_celsius >= 100`

#### 2. Date/Time Operations
Use SQL for temporal calculations and formatting:
- `IsAquarius(dob)` → `strftime('%m%d', dob) BETWEEN '0120' AND '0218'`
- `GetAge(birthdate)` → `strftime('%Y', 'now') - strftime('%Y', birthdate)`
- `FormatDate(date)` → `strftime('%Y-%m-%d', date)`
- `DaysUntilDeadline(deadline)` → `julianday(deadline) - julianday('now')`

#### 3. String Operations
Use SQL for simple string manipulation:
- `FormatPhone(number)` → `CONCAT('(', SUBSTR(number,1,3), ') ', SUBSTR(number,4,3), '-', SUBSTR(number,7,4))`
- `Capitalize(text)` → `UPPER(SUBSTR(text,1,1)) || LOWER(SUBSTR(text,2))`

#### 4. Geographical Information 
Use SQL when mappings are based on well-established, authoritative standards:
- `StateToRegion(state)` → `CASE WHEN state IN ('CA', 'OR', 'WA') THEN 'West' WHEN state IN ('NY', 'MA', 'PA') THEN 'East' ... END`
- `IsAsianCountry(country)` → `country IN ('China', 'Japan', 'India', 'South Korea', 'Thailand', 'Vietnam', 'Indonesia', ...)`

#### 5. LLM-Extracted Facts
Use an LLM **once** to extract factual information, then hard-code it into SQL.
- `IsMostBallonDorWinner(player_name)` → `player_name = 'Lionel Messi'`
- `IsAustrianCapital(city)` → `city = 'Vienna'`

**CRITICAL - Accuracy Requirements:**
- Ensure geographic, historical, or domain-specific claims reflect precise and widely accepted definitions, not approximations or political aggregations.

### Cannot Be Replaced

A UDF **cannot be replaced** when it requires semantic understanding, external data, or subjective judgment

#### 1. Semantic Understanding
- `AnalyzeSentiment(text)` - Needs NLP to determine positive/negative/neutral sentiment
- `ClassifyTopic(text)` - Requires understanding content themes and context
- `SummarizeText(content)` - Requires comprehension and generation
- `AreSimilar(text1, text2)` - Similarity is subjective without clear threshold

#### 2. Complex Domain Knowledge Beyond Simple Rules
- `DiagnoseSeverity(symptoms)` - Medical diagnosis requires expertise beyond simple if/else rules
- `AssessLegalCompliance(contract)` - Legal interpretation is jurisdiction-dependent and nuanced
- `EvaluateProductQuality(description)` - Subjective quality assessment without objective metrics
- `IsAppropriateContent(text, audience)` - Context and cultural considerations vary

'''