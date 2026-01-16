Question_analysis_prompt = '''
# Question Analysis Task

Analyze the given question to generate a query execution instruction.

## Database Metadata
{db_schema}

## Question
{question}

---

## Task Overview

Generate a step-by-step execution plan that identifies:
1. Which operators are needed (SQL and/or LLM)
2. The correct sequence of operations
3. When traditional SQL operation cannot handle the problem and LLM invocations are necessary

---

## Available Operators

### SQL Operations
- selection: Filter rows based on conditions
- projection: Select or compute specific columns  
- join: Combine tables on matching keys
- top_k: Sort and limit results
- aggregation: aggregation and GROUP BY with aggregate functions (COUNT, SUM, AVG, etc.)

**Best Practices:**
- **Always check domain notes first** for formulas, calculations, and business logic definitions
- **Always check the data type and sampled data to generate the correct literals** 
   - Copy literal formats from samples precisely
   - For decimals (0-1 range): use `>= 0.5` for "at least 50%"
   - For integers (0-100 range): use `>= 50` for "at least 50%"
   - For datetime columns: Match the complete format from samples
     * If samples show '2010-07-19 19:15:52.0', use either:
       - `Date >= '2014-09-14 00:00:00.0'` (exact format), OR
       - `Date LIKE '2014-09-14%'` (pattern matching)
   - For unit conversions: when the question uses different units than the database, convert the value in the WHERE clause
- **If a required column doesn't exist, check if it can be derived using LLM operations**

### LLM Operations (Semantic Data Processing, World Knowledge Utilization)
- llm_selection: Filtering or selecting rows based on semantic criteria
  - Use when: Filtering requires semantic understanding, contextual reasoning, or leveraging the LLM's parametric knowledge to evaluate conditions/select rows.
  - **NOT for matching across tables - use llm_join instead**
  - Returns: Subset of rows matching the condition
- llm_projection: Derives NEW COLUMNS through semantic analysis of existing data
  - Use when: Creating new columns by inferring missing attributes, answering row-specific questions
  - Returns: All input rows with additional derived columns
- llm_join: Matches inputs from two tables based on semantic criteria
  - Use when: Direct equivalence is insufficient and matching requires semantic understanding, entity linking, or transformation (e.g., "British" → "UK")
  - Returns: Joined pairs based on semantic matching
- llm_topk: Ranks rows by subjective/semantic criteria
  - Use when: Ranking requires subjective/semantic judgment or qualitative comparison of entities where no objective formula exists
  - Returns: Top-k rows ordered by semantic ranking
- llm_aggregation: Aggregates **multiple rows** and returns insights or summarizations
  - Use when:  Generating summaries, identifying patterns, or providing high-level insights from collections of entities
  - Returns: Aggregate results (summaries, insights)

**Note:** When selecting LLM operators, identify the core operation:
- **Filtering rows** → llm_selection
- **Creating columns** → llm_projection  
- **Matching across tables** → llm_join
- **Ranking rows** → llm_topk
- **Summarizing multiple rows** → llm_aggregation

- **Handling Superlatives ('most', 'least', 'best', 'worst') in the question:**
Determine whether the question requires **searching** for specific values or **comparing** content across rows.
**Use llm_selection** when:
- Question asks to FIND rows that match objective criteria
- Examples: "country with largest GDP", "most popular Spotify artist"
**Use llm_topk** when:
- Question asks to ordering/ranking multiple rows by degree of a quality
- Examples: "most positive review", "least related to topic X", "most controversial statement", "safest city to live"

**Use llm_join directly when:**
- Matching columns from two tables requires semantic similarity, fuzzy matching, or transformation (e.g., country → nationality, team name → country name)
- Checking if content from one table is "related to", "similar to", or "relevant to" content from another table

---

## Robustness Requirements

**ALWAYS Add safety notes to guide implementation. Include notes in the plan ONLY when:**
1. **NULL Filtering for TopK Operations:**
   - Add Note: **Execution Note:** Input column `col` should filter NULL values before ordering
2. **NULL Filtering for Aggregations:**
   - Add Note: **Execution Note:** Input column `col` should filter NULL values before aggregation
3. **Division Safety:**
   - Add Note: **Execution Note:** Input columns should validate: `numerator` IS NOT NULL AND `denominator` IS NOT NULL AND `denominator` != 0
**Rounding to Nearest Integer (When Question Requires):**
   - Add a Projection operator with column: `ROUND(`column_to_round`, 0)`

---

## Output Format

Intent: [What is the user asking for?]

Required Operators: [Mark with [x] ALL operators needed to obtain the final result.]
- [ ] selection
- [ ] projection  
- [ ] join
- [ ] top_k
- [ ] aggregation
- [ ] llm_selection
- [ ] llm_projection
- [ ] llm_join
- [ ] llm_topk
- [ ] llm_aggregation

LLM Operation Justifications:
[For EACH checked LLM operator, explain why SQL is insufficient]

Execution Plan:
[The last step must produce the final output requested in the question.]

---

## Examples

Question: Which schools in Southern California offer a magnet program, and what is their SAT excellence rate rounded to the nearest integer?

Output:
Intent: Identify magnet schools in Southern California and calculate their SAT excellence rates rounded to the nearest integer

Required Operators:
- [x] selection
- [x] projection  
- [x] join
- [x] llm_selection

LLM Operation Justifications:
- llm_selection: Determining whether a county belongs to "Southern California" requires geographic knowledge

Execution Plan:
Step 1: selection - Filter the `schools` table where `Magnet` = 1
Step 2: llm_selection - Filter the table from Step 1 where `County` is in Southern California based on geographic knowledge
Step 3: join - Combine the table from Step 2 with `satscores` on `schools.CDSCode = satscores.cds`
Step 4: projection - Compute the SAT excellence rate for each school as `excellence_rate = ROUND(NumGE1500 / NumTstTakr, 0)` from the table in Step 3
   - **Execution Note:** Input columns should validate: `NumGE1500` IS NOT NULL AND `NumTstTakr` IS NOT NULL AND `NumTstTakr` != 0
Final Output: School, excellence_rate (row set)

---
Question: List the home countries of the top five finishers in 2008 Malaysian Grand Prix and rank these countries by their global influence from highest to lowest.

Output:
Intent: Identify the top 5 finishers in the 2008 Malaysian Grand Prix, extract their home countries, and rank those countries by global influence

Required Operators:
- [x] selection
- [x] join
- [x] top_k
- [x] llm_projection
- [x] llm_topk

LLM Operation Justifications:
- llm_projection: Converting `nationality` values (e.g., "British", "German") to country names requires semantic understanding and knowledge mapping
- llm_topk: Ranking countries by "global influence" is a subjective assessment requiring world knowledge and qualitative judgment

Execution Plan:
Step 1: selection - Filter the `races` table where `year` = 2008 AND `name` = 'Malaysian Grand Prix'
Step 2: join - Combine the table from Step 1 with `results` on `races.raceId = results.raceId`
Step 3: top_k - Rank by `position` in ascending order in the table from Step 2, LIMIT 5
   - **Execution Note:** Input columns `position` should filter NULL values before ordering
Step 4: join - Combine the table from Step 3 with `drivers` on `results.driverId = drivers.driverId`
Step 5: llm_projection - Extract the `country` from `nationality` in the table from Step 4
Step 6: llm_topk - Rank `country` by global influence in the table from Step 5
Final Output: country (row set)

---
Question: How many authors in our database have published books with publishers located in their home country?

Output:
Intent: Count the number of authors who have published books with publishers based in the same country as the author

Required Operators:
- [x] join
- [x] aggregation
- [x] llm_join

LLM Operation Justifications:
- llm_join: Matching author nationality (e.g., "American", "French") with publisher location (e.g., "USA", "France") requires semantic transformation. Direct string matching fails because "American" ≠ "USA" and "British" ≠ "United Kingdom".

Execution Plan:
Step 1: join - Combine the `authors` table with `books` on `authors.author_id = books.author_id`
Step 2: join - Combine the table from Step 1 with `publishers` on `books.publisher_id = publishers.publisher_id`
Step 3: llm_join - Match `nationality` with `location` from Step 2 based on semantic equivalence (e.g., "American" → "USA", "French" → "France")
   - **Execution Note:** Ensure both columns are non-null before joining
Step 4: aggregation - COUNT DISTINCT `author_id` from the table in Step 4
Final Output: count (scalar)

---
Question: Summarize the feature of powers of female superherors from DC Cosmics?

Output:
Intent: Provide a summary of the characteristics and features of powers possessed by female DC Comics superheroes

Required Operators:
- [x] selection
- [x] join
- [x] llm_aggregation

LLM Operation Justifications:
- llm_aggregation: Summarizing power features requires synthesizing information across multiple rows and generating natural language descriptions

Execution Plan:
Step 1: selection - Filter the `gender` table where `gender` = 'Female'
Step 2: selection - Filter the `publisher` table where `publisher_name` = 'DC Comics'
Step 3: join - Combine the table from Step 1, the table from Step 2, and `superhero` on `superhero.gender_id = gender.id` AND `superhero.publisher_id = publisher.id`
Step 4: join - Combine the table from Step 3 with `hero_power` on `hero_power.hero_id = superhero.id`
Step 5: join - Combine the table from Step 4 with `superpower` on `hero_power.power_id = superpower.id`
Step 6: llm_aggregation - Summarize the features of the powers based on `power_name` in the table from Step 5
Final Output: power_name, power_features (row set)

---
Question: How many schools have an average SAT score higher than the average SAT score across all U.S. students in 2020?

Output:
Intent: Count the number of schools whose average SAT scores exceed the national average SAT score for 2020

Required Operators:
- [x] projection  
- [x] aggregation
- [x] llm_selection

LLM Operation Justifications:
- llm_selection: Comparing against "the average SAT score across all U.S. students in 2020" requires external knowledge of the national average

Execution Plan:
Step 1: projection - Compute `SchoolAvgSATScore` as `(AvgScrRead + AvgScrMath + AvgScrWrite)` from the `satscores` table
Step 2: llm_selection - Filter schools where the computed average score is greater than the national average SAT score for 2020 in the table from Step 1
Step 3: aggregation - COUNT the number of schools in the table from Step 2
Final Output: count (scalar)

---

**Question:** "What are the top 5 most family-friendly restaurants in downtown locations?"

Output:
**Intent:** Identify restaurants in downtown areas and rank them by how family-friendly they are.

**Required Operators:**
- [x] selection
- [x] llm_topk

**LLM Operation Justifications:**
- `llm_topk`: "Family-friendly" is a subjective, multi-dimensional quality involving atmosphere, menu options, noise level, and amenities. SQL cannot assess or rank these qualitative attributes. The LLM must evaluate and rank all downtown restaurants by family-friendliness.

**Execution Plan**
Step 1: selection - Filter the `restaurants` table where `location` contains "downtown"
Step 2: llm_topk - Rank all downtown restaurants by family-friendliness, LIMIT 5
Final Output: restaurant_name (5 rows)

---

Question: Which product categories have total sales greater than $100,000?

Output:
Intent: Identify product categories exceeding $100K in total sales

Required Operators:
- [x] aggregation
- [x] selection

LLM Operation Justifications:
[None - all operations can be handled with SQL]

Execution Plan:
Step 1: aggregation - Compute `total_sales` as SUM(`sales_amount`) grouped by `category` from the `sales` table
   - **Execution Note:** Input column `sales_amount` should filter NULL values before aggregation
Step 2: selection - Filter the table from Step 1 where `total_sales` > 100000
Final Output: category, total_sales (row set)

'''


NL2SQL_instruct = ''' 

# Query Plan Generation Task

You are constructing a JSON query plan from a user question using the provided database metadata and execution plan.

## Question
{question}

## Database Metadata
{db_schema}

## Execution Plan
{Hint}

---

# Instructions

## Overview
Each query plan is a tree of computational nodes. Each node represents either:
- **SQL Operation**: Standard database operations (filter, join, aggregate, etc.)
- **LLM Operation**: AI-powered semantic processing via User-Defined Functions (UDFs)

## Node Templates

### SQL Operations (Standard Database Operations)

(1) Table Node: Reference an input table
{{
  "Node Type": "Table",
  "Table Name": ""
}}

(2) Selection:  Filter rows based on a condition
{{
  "Node Type": "Selection",
  "Condition": "col = val",
  "Input": {{ ... }}
}}

(3) Projection: Select or compute output columns or apply build-in functions
{{
  "Node Type": "Projection",
  "Columns": ["a", "b", "c = a * b", "ROUND(c,0)"],
  "Distinct": true/false,    //optional
  "Input": {{ ... }}
}}

(4) Join: Connect two inputs with a condition
{{
  "Node Type": "Join",
  "Condition": "T1.col = T2.col",
  "Inputs": [{{ ... }}, {{ ... }}]
}}

(5) TopK: Select top-ranked rows
{{
  "Node Type": "TopK",
  "k": int or inf,
  "Ranking criteria": "col1 asc/desc, col2 asc/desc",
  "Input": {{ ... }}
}}

(6) Aggregation: Apply aggregation with or without grouping
{{
  "Node Type": "Aggregation",
  "Aggregate Function": [{{
    "Function Name": "",
    "Distinct": true/false,    //optional
    "Input Columns": [],
    // Use CASE expressions in Input Columns to compute conditional aggregations
    // Example: ["CASE WHEN gender = 'Female' THEN score END"] 
    // This allows aggregating different subgroups in a single aggregation node
    "Output Column": ""
  }}
  // ... multiple functions supported per node
  ],
  "GroupBy Columns": [], // optional: include for per-group results, omit for overall totals or when filtering to specific groups first
  "Input": {{ ... }}
}}

**Deduplication in Projection and Aggregation:**
- Set "Distinct": true when counting/projecting uniqueness

### LLM Operations (UDF-Enhanced for Semantic Processing)

(1) Selection: Filtering based on semantic conditions
{{
  "Node Type": "Selection",
  "UDF": {{
    "UDF Name": "",
    "Input Columns": []
  }},
  "Input": {{ ... }}
}}

(2) Projection:  Extracts information from each row via semantic analysis
{{
  "Node Type": "Projection",
  "UDF": {{
    "UDF Name": "",
    "Input Columns": [],
    "Output Column": ""
  }},
  "Input": {{ ... }}
}}

(3) Join: Matches inputs from two tables based on semantic criteria
{{
  "Node Type": "Join",
  "UDF": {{
    "UDF Name": "",
    "Input Columns": [[left_table_join_keys], [right_table_join_keys]]
  }},
  "Inputs": [{{ ... }}, {{ ... }}] 
}}

(4) Topk: Ranks rows by subjective/semantic criteria
{{
  "Node Type": "TopK",
  "k": int or inf,
  "UDF": {{
    "UDF Name": "", 
    "Input Columns": []
  }}, 
  "Input": {{ ... }}
}}

(5) Aggregation: Aggregates multiple rows and returns insights or summarizations
{{
  "Node Type": "Aggregation",
  "UDF": {{
    "UDF Name": "",
    "Input Columns": [],
    "Output Column": ""
  }},
  "GroupBy Columns": [],  // optional (omit if no grouping)
  "Input": {{ ... }}
}}

### **Important**: UDF Design Rules
1. **Choose input columns that provide necessary context for the LLM**: 
   - "Is this school in Southern California?" → ✅ `county_name` (contains location) ❌ `sat_score` (unrelated)
   - "What is the most sarcastic post?" → ✅ `post_body` (LLM needs content to judge tone) ❌ `post_id`
   
   **When attributes needed for reasoning are unavailable, use entity identifiers (names/IDs) that enable LLM knowledge retrieval**
      - Example Question: "Which drivers debuted before Lewis Hamilton?"
        - ✅ CORRECT: `DebutedBeforeLewisHamilton(driver_name)`
          - Why: Debut year isn't in the database, but LLM knows driver career timelines from names
        - ❌ WRONG: `DebutedBeforeLewisHamilton(dob)` 
        
2. **Name UDFs as functions**: Create clear, precise names that specify the purpose and avoid ambiguity.
   - Good: `"IsTallerThanBillClinton"` (selection), `"ExtractYearsOfExperience"` (projection), `"MostCreativeName"` (topk), `"SummarizePowers"` (aggregation)
   - Bad: `"Process"`, `"Filter"`, `"Summarize"`

3. **Add format suffixes when output format is specified**: If the question requires specific formatting, append it to the UDF name
   - Example: `"DetermineEligibility_AnswerWithYesOrNo"`

## Plan Construction Workflow
1. **Build the node tree**
   - Follow the execution strategy provided in the execution plan, **ALWAYS incorporate execution notes**:
     - Use CASE expressions for conditional logic (e.g., null handling: "CASE WHEN col IS NULL THEN 1 ELSE 0 END")
  - Select appropriate node types
      - Use SQL templates for exact matching, arithmetic, standard aggregations
      - Use UDF templates for semantic filtering, text extraction, fuzzy matching, subjective ranking, or text summarization
   - Start from the innermost node (usually Table nodes)
   - Work outward, adding operations layer by layer
   - Verify that all input columns for each node exist in its upstream node(s). 
   - **When the same column name appears in multiple tables in schema, you MUST disambiguate by prefixing with the table name using dot notation: `table_name.column_name`** Especially for id or name columns, (e.g., if both `races` and `drivers` tables have a `name` column, specify `drivers.name` or `races.name` rather than just `name` to avoid ambiguity)
   - When there is an id column, you MUST specify its table prefix
   - Project the final result as asked in the question


2. **Validate the Final Plan**
   **Column Validation:**
   - Include ONLY the explicitly requested columns in the final projection
   **Syntax & Structure:**
   - Ensure valid JSON syntax (properly matched brackets and braces)
   - Verify operators are in the correct execution order
   **UDF Placement:**
   - UDFs must be fields within nodes, NOT standalone nodes

---

## Examples:
{few_shot}

'''

NL2SQL_few_shot_examples = '''

Question: Which schools in Southern California offer a magnet program, and what is their SAT excellence rate rounded to the nearest integer?

Database Metadata:
Table: schools
Columns: ['County', 'Magnet', 'CDSCode', 'School']

Table: satscores
Columns: ['cds', 'NumGE1500','NumTstTakr']

Execution Plan:
Step 1: selection - Filter the `schools` table where `Magnet` = 1
Step 2: llm_selection - Filter the table from Step 1 where `County` is in Southern California based on geographic knowledge
Step 3: join - Combine the table from Step 2 with `satscores` on `schools.CDSCode = satscores.cds`
Step 4: projection - Compute the SAT excellence rate for each school as `excellence_rate = ROUND(NumGE1500 / NumTstTakr, 0)` from the table in Step 3
   - **Execution Note:** Input columns should validate: `NumGE1500` IS NOT NULL AND `NumTstTakr` IS NOT NULL AND `NumTstTakr` != 0
Final Output: School, excellence_rate (row set)

Output: 
{{
  "Node Type": "Projection",
  "Columns": [
    "School", 
    "CASE WHEN NumGE1500 IS NOT NULL AND NumTstTakr IS NOT NULL AND NumTstTakr != 0 THEN ROUND(NumGE1500 / NumTstTakr, 0) END AS ExcellenceRate"
  ],
  "Distinct": true,
  "Input": {{
    "Node Type": "Join",
    "Condition": "satscores.cds = schools.CDSCode",
    "Inputs": [
      {{
        "Node Type": "Selection",
        "UDF": {{
          "UDF Name": "isInSouthernCalifornia",
          "Input Columns": ["County"]
        }},
        "Input": {{
          "Node Type": "Selection",
          "Condition": "Magnet = 1",
          "Input": {{ "Node Type": "Table", "Table Name": "schools" }}
        }}
      }},
      {{ "Node Type": "Table", "Table Name": "satscores" }}
    ]
  }}
}}

---

Question: List the home countries of the top five finishers in 2008 Malaysian Grand Prix and rank these countries by their global influence from highest to lowest.

Database Metadata:
Table: races
Columns: ['raceId', 'year', 'name']

Table: results
Columns: ['raceId', 'position', 'driverId']

Table: drivers
Columns: ['driverId', 'nationality']

Execution Plan:
Step 1: selection - Filter the `races` table where `year` = 2008 AND `name` = 'Malaysian Grand Prix'
Step 2: join - Combine the table from Step 1 with `results` on `races.raceId = results.raceId`
Step 3: top_k - Rank by `position` in ascending order in the table from Step 2, LIMIT 5
   - **Execution Note:** Input columns `position` should filter NULL values before ordering
Step 4: join - Combine the table from Step 3 with `drivers` on `results.driverId = drivers.driverId`
Step 5: llm_projection - Extract the `country` from `nationality` in the table from Step 4
Step 6: llm_topk - Rank `country` by global influence in the table from Step 5
Final Output: country (row set)

Output: 
{{
  "Node Type": "TopK",
  "k": "inf",
  "UDF": {{
    "UDF Name": "MostGloballyInfluentialCountry",
    "Input Columns": ["country"]
  }},
  "Input": {{
    "Node Type": "Projection",
    "UDF": {{
      "UDF Name": "ExtractCountryFromNationality",
      "Input Columns": ["nationality"],
      "Output Column": "country"
    }},
    "Input": {{
      "Node Type": "Projection",
      "Columns": ["nationality"],
      "Distinct": true,
      "Input": {{
        "Node Type": "Join",
        "Condition": "results.driverId = drivers.driverId",
        "Inputs": [
          {{
            "Node Type": "TopK",
            "k": 5,
            "Ranking criteria": "CASE WHEN position IS NULL THEN 1 ELSE 0 END, position ASC",
            "Input": {{
              "Node Type": "Join",
              "Condition": "races.raceId = results.raceId",
              "Inputs": [
                {{
                  "Node Type": "Selection",
                  "Condition": "year = 2008 AND name = 'Malaysian Grand Prix'",
                  "Input": {{ "Node Type": "Table", "Table Name": "races" }}
                }},
                {{ "Node Type": "Table", "Table Name": "results" }}
              ]
            }}
          }},
          {{ "Node Type": "Table", "Table Name": "drivers" }}
        ]
      }}
    }}
  }}
}}

---

Question: How many authors in our database have published books with publishers located in their home country?

Database Metadata:
Table: authors
Columns: ['author_id', 'author_name', 'nationality']

Table: books
Columns: ['book_id', 'author_id', 'publisher_id', 'title']

Table: publishers
Columns: ['publisher_id', 'publisher_name', 'location']

Execution Plan:
Step 1: join - Combine the `authors` table with `books` on `authors.author_id = books.author_id`
Step 2: join - Combine the table from Step 1 with `publishers` on `books.publisher_id = publishers.publisher_id`
Step 3: llm_join - Match `nationality` with `location` from Step 2 based on semantic equivalence (e.g., "American" → "USA", "French" → "France")
   - **Execution Note:** Ensure both columns are non-null before joining
Step 4: aggregation - COUNT DISTINCT `author_id` from the table in Step 4
Final Output: count (scalar)

Output: 
{{
  "Node Type": "Aggregation",
  "Aggregate Function": [{{
    "Function Name": "COUNT",
    "Distinct": true,
    "Input Columns": ["author_id"],
    "Output Column": "count"
  }}],
  "Input": {{
    "Node Type": "Join",
    "UDF": {{
      "UDF Name": "isNationalitySameAsCountry",
      "Input Columns": [["authors.nationality"], ["publishers.location"]]
    }},
    "Inputs": [
      {{
        "Node Type": "Join",
        "Condition": "books.publisher_id = publishers.publisher_id",
        "Inputs": [
          {{
            "Node Type": "Join",
            "Condition": "authors.author_id = books.author_id",
            "Inputs": [
              {{ "Node Type": "Table", "Table Name": "authors" }},
              {{ "Node Type": "Table", "Table Name": "books" }}
            ]
          }},
          {{"Node Type": "Table", "Table Name": "publishers"}}
        ]
      }},
      {{"Node Type": "Table", "Table Name": "publishers"}}
    ]
  }}
}}

---

Question: Summarize the feature of powers of female superherors from DC Cosmics?

Database Metadata:
Table: gender
Columns:['id', 'gender']

Table: publisher
Columns: ['id','publisher_name']

Table: superhero
Columns: ['id','gender_id','publisher_id']

Table: hero_power
Columns:['hero_id','power_id']

Table: superpower
Columns:['id','power_name']

Execution Plan:
Step 1: selection - Filter the `gender` table where `gender` = 'Female'
Step 2: selection - Filter the `publisher` table where `publisher_name` = 'DC Comics'
Step 3: join - Combine the table from Step 1, the table from Step 2, and `superhero` on `superhero.gender_id = gender.id` AND `superhero.publisher_id = publisher.id`
Step 4: join - Combine the table from Step 3 with `hero_power` on `hero_power.hero_id = superhero.id`
Step 5: join - Combine the table from Step 4 with `superpower` on `hero_power.power_id = superpower.id`
Step 6: llm_aggregation - Summarize the features of the powers based on `power_name` in the table from Step 5
Final Output: power_name, power_features (row set)

Result: 
{{
  "Node Type": "Projection",
  "Columns": ["power_name", "power_features"],
  "Distinct": true,
  "Input": {{
    "Node Type": "Aggregation",
    "UDF": {{
      "UDF Name": "summarizePowers",
      "Input Columns": ["power_name"],
      "Output Column": "power_features"
    }},
    "Input": {{
      "Node Type": "Join",
      "Condition": "hero_power.hero_id = superhero.id",
      "Inputs": [
        {{
          "Node Type": "Join",
          "Condition": "superhero.gender_id = gender.id",
          "Inputs": [
            {{
              "Node Type": "Join",
              "Condition": "superhero.publisher_id = publisher.id",
              "Inputs": [
                {{
                  "Node Type": "Selection",
                  "Condition": "gender.gender = 'Female'",
                  "Input": {{
                    "Node Type": "Table",
                    "Table Name": "gender"
                  }}
                }},
                {{
                  "Node Type": "Selection",
                  "Condition": "publisher.publisher_name = 'DC Comics'",
                  "Input": {{ "Node Type": "Table", "Table Name": "publisher" }}
                }}
              ]
            }},
            {{ "Node Type": "Table", "Table Name": "superhero" }}
          ]
        }},
        {{
          "Node Type": "Join",
          "Condition": "hero_power.power_id = superpower.id",
          "Inputs": [
            {{ "Node Type": "Table", "Table Name": "hero_power" }},
            {{ "Node Type": "Table", "Table Name": "superpower" }}
          ]
        }}
      ]
    }}
  }}
}}

---

Question: How many schools have an average SAT score higher than the average SAT score across all U.S. students in 2020?

Database Metadata:
Table: satscores
Columns: ['cds', 'AvgScrRead', 'AvgScrMath', 'AvgScrWrite']

Execution Plan:
Step 1: projection - Compute `SchoolAvgSATScore` as `(AvgScrRead + AvgScrMath + AvgScrWrite)` from the `satscores` table
Step 2: llm_selection - Filter schools where the computed average score is greater than the national average SAT score for 2020 in the table from Step 1
Step 3: aggregation - COUNT the number of schools in the table from Step 2
Final Output: count (scalar)

Output:
{{
  "Node Type": "Aggregation",
  "Aggregate Function": [{{
      "Function Name": "COUNT",
      "Distinct": true,
      "Input Columns": ["cds"],
      "Output Column": "NumOfSchoolsAboveUSSATAvg"
    }}],
  "Input": {{
    "Node Type": "Selection",
    "UDF": {{
      "UDF Name": "isHigherThanUSSATAverageIn2020",
      "Input Columns": ["SchoolAvgSATScore"]
    }},
    "Input": {{
      "Node Type": "Projection",
      "Columns": [
        "CDSCode",
        "SchoolAvgSATScore = AvgScrRead + AvgScrMath + AvgScrWrite"
      ],
      "Input": {{ "Node Type": "Table", "Table Name": "satscores" }}
    }}
  }}
}}
'''
