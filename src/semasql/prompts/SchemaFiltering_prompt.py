FilterSchema_reasoning_prompt = '''# Column Selection Task

Select the minimal set of tables, columns, relationships, and domain notes needed to answer the question. Remove all irrelevant information.

## Database Metadata
{DATABASE_SCHEMA}

## Question
{QUESTION}

---

## Instructions

**Step 1: Analyze the question**
First, think through:
- What columns are needed for filtering/calculations/knowledge retrieval/output? 
- What columns represent the entities/content mentioned in the query?
  Include human-readable columns to enable LLM reasoning/knowledge extraction: 
  - Entity names: `player.name`, `driver.name`, team.team_name`, `company.name`,`product.product_name`, `book.title`
  - Content/text: `post.text`, `comment.body`, `book.description`
  - **NOT numeric IDs** (e.g., `player_id`, `post_id`) unless needed for:
    - Joins: connecting tables
    - Counts: counting distinct entities
- What tables contain these columns?

**Step 2: Identify join paths**
- What tables need to be connected?
- Are bridge tables needed?
- What are the foreign key relationships?

**Step 3: Select relevant domain notes**
- Which domain notes explain terminology used in the question?
- Which domain notes clarify how to interpret or use the selected columns?

**Step 4: Verify completeness**
- For each relationship, are BOTH the FK and referenced PK kept?
- Are all columns mentioned in domain notes included?
- Are only elements that directly help answer the question kept?

---

## Process

1. **Identify core requirements**: What columns answer the question directly?
2. **Check domain notes**: Are there predefined formulas or rules that apply?
3. **Add context**: What entity identifiers or content fields are needed in LLM reasoning?
4. **Map relationships**: What tables must join, and through what keys?
5. **Verify completeness**: Is each FK paired with its PK? Are relevant domain notes included?

---

## Output Format

Provide your response in TWO sections, **do not drop the XML labels** (<reasoning> and <filtered_schema>):

### REASONING
<reasoning>
[Include a concise step-by-step analysis here]
</reasoning>

### FILTERED_SCHEMA
<filtered_schema>
[Return ONLY the selected schema elements in their EXACT original YAML format.]
Required structure [Use ACTUAL names from the input schema]:
name:
descriptions:
tables:
  - name: table_name
    description: table description
    columns:
      - name: column_name
        description: column description
        data type: TYPE
        data description: detailed description
        data samples: [sample1, sample2, sample3]
relationships:
  - name: relationship_name
    left table: table1
    right table: table2
    left column: column1
    right column: column2
domain notes:
  - note text
</filtered_schema>

'''
