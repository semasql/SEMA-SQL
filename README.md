# SEMA-SQL: Beyond Traditional Relational Querying with Large Language Models

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

**Sema-SQL** is an innovative framework that seamlessly integrates Large Language Models (LLMs) with relational databases. It extends the capabilities of traditional SQL by embedding AI-powered semantics into the query pipeline, allowing users to ask complex, real-world questions in natural language.

## 📖 Introduction

Relational databases offer an efficient, accurate, and scalable platform for storing and querying structured data. However, real-world user questions often transcend the capabilities of traditional database models. For example, a query might require:

*   **External World Knowledge** (e.g., "Which circuits are in Europe?")
*   **Semantic Understanding of Text** (e.g., "Find all reviews that mention 'performance issues'")
*   **Complex Reasoning** (e.g., "Categorize products based on their descriptions")

**SEMA-SQL** is designed to bridge this gap. It formally integrates LLM semantics into relational algebra, enabling powerful new operators that connect structured data querying with broader, knowledge-based reasoning. It is a fully automated system that embeds LLMs into the query pipeline to answer natural language questions grounded in structured data, while also supporting text comprehension and leveraging external world knowledge.


## 🚀 Getting Started

### 1. Setup Environment

Install all required dependencies:

```shell
pip install -r requirements.txt
```

### 2. Prepare Data
Before evaluation, download and prepare the data and query files. The database file used for evaluation is a subset of the [BIRD benchmark](https://bird-bench.github.io/), which can be downloaded from the [data repository](https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip). We expect the unzipped database file to be stored in the project root directory (e.g., `data/MINIDEV/dev_databases/...`).


### 3. Configure API Keys

Set up your API keys as environment variables. Create a `.env` file in the project root:

```shell
# Example .env file
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=your_base_url_here
QWEN_API_KEY=your_qwen_api_key_here  # If using Qwen model
```

Supported models: `gpt`, `claude`, `qwen`, `gemini`

### 4. Run Evaluation

Ensure your current working directory is the project root, `SEMA-SQL`.

*   **Single Query Evaluation**:
    ```shell
    # Run the pipeline for a single predefined question
    ./evaluation.sh
    ```
    
    You can customize the query by passing arguments:
    ```shell
    ./evaluation.sh \
        --db_name european_football_2 \
        --query "Your natural language question here" \
        --llm_model gpt
    ```
    
    Available options:
    - `--db_path`: Path to database directory (default: `data/MINIDEV/dev_databases`)
    - `--db_name`: Database name (default: `california_schools`)
    - `--query`: Natural language query to execute
    - `--llm_model`: LLM model to use (`gpt`, `claude`, `qwen`, `gemini`)
    - `--column_selector`: Enable schema filtering (1) or disable (0)
    - `--question_decomposition`: Enable question decomposition (1) or disable (0)
    - `--optimization_lazy_llm`: Enable UDF deferral optimization (1) or disable (0)
    - `--optimization_udf_rewrite`: Enable UDF rewriting optimization (1) or disable (0)
    

## ⚙️ How It Works: An Example

Let's walk through how SEMA-SQL processes a query requiring external knowledge. We'll use the example query from `evaluation.sh`:

**Natural Language Input**:
> "Among the schools with the average score in Math over 560 in the SAT test, how many schools are in the bay area?"

The database contains `County` and `City` columns, but doesn't explicitly identify which locations belong to the "Bay Area". SEMA-SQL leverages an LLM to infer this geographic knowledge.

---

### Schema Generation

Before querying, SEMA-SQL requires a semantic data model that describes the database schema. The `generate_db_schema` function in `schema_generator.py` automatically generates this YAML schema document by combining database metadata (table structures, column types) with CSV-based column descriptions, using an LLM to create comprehensive, human-readable schema documentation. This generated schema enables semantic understanding of the database structure for subsequent query processing.

### Schema Filtering

SEMA-SQL first filters the database schema to retain only information relevant to the question:

```
The concise schema is
['schools.CDSCode', 'schools.County', 'schools.School', 'schools.City', 
 'satscores.cds', 'satscores.AvgScrMath']
```

This reduces the schema from all available columns to only those needed for the query.

---

### Question Decomposition & Plan Generation

SEMA-SQL analyzes the question and generates a logical query plan:

**Question Analysis:**
- **Intent:** Count the number of schools located in the Bay Area that have an average SAT Math score exceeding 560.
- **Required Operators:** selection, join, aggregation, llm_selection
- **LLM Operation Justification:** Determining whether a county or city belongs to the "Bay Area" requires geographic knowledge that cannot be determined through SQL pattern matching alone.

**Generated Query Plan (JSON):**

```json
{
  "Node Type": "Aggregation",
  "Aggregate Function": [
    {
      "Function Name": "COUNT",
      "Distinct": true,
      "Input Columns": ["schools.CDSCode"],
      "Output Column": "count"
    }
  ],
  "Input": {
    "Node Type": "Selection",
    "UDF": {
      "UDF Name": "isInBayArea",
      "Input Columns": ["County"]
    },
    "Input": {
      "Node Type": "Join",
      "Condition": "satscores.cds = schools.CDSCode",
      "Inputs": [
        {
          "Node Type": "Selection",
          "Condition": "AvgScrMath > 560",
          "Input": {
            "Node Type": "Table",
            "Table Name": "satscores"
          }
        },
        {
          "Node Type": "Table",
          "Table Name": "schools"
        }
      ]
    }
  }
}
```

---

### Query Optimization

SEMA-SQL's optimizer attempts to rewrite the semantic UDF into efficient SQL. In this case, the optimizer recognizes that `isInBayArea` can be transformed into a SQL `IN` clause:

**Optimizer Output:**
```json
{
  "result": "County IN ('Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco', 'San Mateo', 'Santa Clara', 'Solano', 'Sonoma')",
  "explanation": "The San Francisco Bay Area is a well-defined metropolitan region in Northern California consisting of 9 counties according to the Association of Bay Area Governments (ABAG) and U.S. Census Bureau definitions..."
}
```

---

###  Query Execution

The final, executable SQL is generated:

```sql
SELECT COUNT(DISTINCT isInBayArea_input_table."CDSCode") AS count
FROM isInBayArea_input_table
WHERE isInBayArea_input_table."County" IN (
    'Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco', 
    'San Mateo', 'Santa Clara', 'Solano', 'Sonoma'
);
```

**Final Result:** `[(71,)]`

This example perfectly illustrates how SEMA-SQL translates a natural language question requiring geographic knowledge into a precise, optimized, and executable SQL query enriched with LLM-driven semantics. The system automatically:
1. Filters the schema to relevant columns
2. Decomposes the question and generates a logical plan
3. Optimizes the plan by rewriting UDFs to efficient SQL when possible
4. Executes the query and returns the result

## 📊 Evaluation & Benchmarks

To assess the effectiveness of our framework, we developed a comprehensive benchmark.

### Datasets
We use a subset of the [BIRD benchmark](https://bird-bench.github.io/) as the backend datasets.


### Benchmark: TAG & TAG-Plus
We evaluated Sema-SQL on the [TAG benchmark](https://github.com/TAG-Research/TAG-Bench).

*   **TAG**: This benchmark contains 80 modified questions from BIRD, spanning 5 domains. It is designed to test two core capabilities:
    *   40 questions requiring **parametric knowledge** from LLMs.
    *   40 questions requiring **semantic reasoning** over textual columns.
    *   The original TAG queries are available in `queries/tag_queries.csv`.

*   **TAG-Plus (Our Extension)**:
    We identified two limitations in TAG: (1) it covers only basic question types, and (2) its queries are structurally simple, involving only a single semantic operator. To address this, we created **TAG-Plus** by adding 40 more complex queries. These new queries introduce advanced reasoning patterns (e.g., computational reasoning, semantic categorization) and more complex query structures (e.g., subqueries, multi-hop reasoning).
    *   The extended queries are available in `queries/tag_plus_queries.csv`.
  
## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).