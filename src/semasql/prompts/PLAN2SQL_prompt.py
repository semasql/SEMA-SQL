PLAN2SQL_gen_table_instruct = '''
## Task
Convert a query execution plan (in JSON format with potential LLM UDFs) into executable SQLite SQL code that creates a new table.

## UDF Handling
1. When the query plan contains UDFs, handle them by either:
- Replacing them with equivalent relational expressions, or
- Using the corresponding extended tables provided in the schema

---

## Input:

Query Execution Plan:
{table_json}

Extended Schema:
{extended_table}

Equivalent Relational Expression:
{primitive}

DB Schema: {db_schema}

Table Name: {new_table_name}

---

## Output Requirements

### Structure
1. Generate ONLY the final executable SQL statement with no explanations, comments, or markdown formatting
2. Use the syntax: CREATE TEMP TABLE [Table Name] AS SELECT ...

### Column Usage and Validation
1. Before using any column in WHERE, JOIN, or any other clause, verify which table it belongs to by checking the DB Schema
2. Always use the correct table alias when referencing a column - match it to the table shown in the DB Schema
3. If a column name appears in the query plan conditions but isn't prefixed with a table name, look it up in the DB Schema to determine its actual table ownership

### Column Projection
1. Look at the DB Schema to determine which columns to project

### Naming Conventions
1. Always use table aliases for all tables (e.g., FROM drivers AS d)
2. Always qualify all column references with their table alias (e.g., d."column_name")
3. Enclose all column names in double quotes (e.g., "column_name")
4. ONLY when the DB schema contains columns with the EXACT same name from different tables, use column aliases with table name to distinguish them (e.g., d."status" AS "driver_status", r."status" AS "race_status"). 
1. **CRITICAL: The output table name in CREATE TEMP TABLE must EXACTLY match the "Table Name" provided in the input**
---

## Examples:

Query Execution Plan:
{{
    "Node Type": "Selection",
    "UDF": {{
    "UDF Name": "isF1WorldChampionCountry",
    "Input Columns": ["nationality"]
    }},
    "Input": {{ "Node Type": "Table", "Table Name": "drivers" }}
}}

Extended Schema:
Table: isF1WorldChampionCountry_result_of_drivers
nationality,,TEXT,,,['British', 'German', 'Spanish']
isF1WorldChampionCountry,,TEXT,isF1WorldChampionCountry is a predicate derived from the columns ([nationality]) in Table drivers. You can use [nationality] to join with Table drivers,1 means the selection predicate is satisfied and 0 means that the selection predicate is not satisfied,['1', '0']

DB Schema: ["drivers.driverId", "drivers.forename", "drivers.surname", "drivers.nationality"]

Table Name: DriversFromF1WorldChampionCountry
    
Output:
CREATE TEMP TABLE DriversFromF1WorldChampionCountry AS
SELECT 
    d."driverId",
    d."forename",
    d."surname",
    d."nationality"
FROM drivers AS d
JOIN isF1WorldChampionCountry_result_of_drivers AS r ON d."nationality" = r."nationality"
WHERE r."isF1WorldChampionCountry" = '1';
---------------
Query Execution Plan:
{{  
    "Node Type": "Projection",
    "Columns": ["power_name"]
    "Distinct": false,
    "Input":
      {{
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
                    "Input": {{
                        "Node Type": "Table",
                        "Table Name": "publisher"
                    }}
                    }}
                ]
                }},
                {{
                "Node Type": "Table",
                "Table Name": "superhero"
                }}
            ]
            }},
            {{
            "Node Type": "Join",
            "Condition": "hero_power.power_id = superpower.id",
            "Inputs": [
                {{
                "Node Type": "Table",
                "Table Name": "hero_power"
                }},
                {{
                "Node Type": "Table",
                "Table Name": "superpower"
                }}
            ]
            }}
        ]
      }}
}}

DB Schema: ["superhero.id", "superhero.gender_id", "superhero.publisher_id", "gender.id", "gender.gender", "publisher.id", "publisher.publisher_name", "hero_power.hero_id", "hero_power.power_id", "superpower.id", "superpower.power_name"]

Table Name: SummarizeSuperPower_intput_table

Output:
CREATE TEMP TABLE SummarizeSuperPower_intput_table AS
SELECT 
    sp."power_name"
FROM superhero AS s
JOIN gender AS g ON s."gender_id" = g."id"
JOIN publisher AS p ON s."publisher_id" = p."id"
JOIN hero_power AS hp ON s."id" = hp."hero_id"
JOIN superpower AS sp ON hp."power_id" = sp."id"
WHERE g."gender" = 'Female' AND p."publisher_name" = 'DC Comics';
---------------
Query Execution Plan:
{{
  "Node Type": "Join",
  "Condition": "races.year = seasons.year",
  "Inputs": [
    {{
      "Node Type": "Table",
      "Table Name": "races"
    }},
    {{
      "Node Type": "Table",
      "Table Name": "seasons"
    }}
  ]
}}

DB Schema: ["races.year", "races.name", "seasons.year", "seasons.url"]

Table Name: isRaceCountryMatchNationality_left_table

Output:
CREATE TEMP TABLE isRaceCountryMatchNationality_left_table AS
SELECT 
    r."year" AS "race_year",
    r."name",
    s."year" AS "season_year",
    s."url"
FROM races AS r
JOIN seasons AS s ON r."year" = s."year";
'''

PLAN2SQL_instruct = '''
## Task
Given a query represented as an execution plan in JSON format containing LLM UDFs, generate a valid, executable SQL statement for SQLite.

## Guidelines:

1. UDF Handling:
- For LLM UDFs in the plan, either:
  (a) Replace them with equivalent relational expressions from the "equivalent relational expression" section
  (b) Execute them using the corresponding extended tables from the "extended schema" section

2. SQL Requirements:
- All referenced tables must exist in the provided database schema or extended schema.
- All columns used in any clause (SELECT, WHERE, GROUP BY, ORDER BY, JOIN) must be present in the schema or derived from valid subqueries.
- The generated SQL strictly follow the query plan except for the handle of UDFs.
- All columns in the SQL query must be explicitly qualified with their respective table names (e.g., table_name.column_name) to avoid OperationalError: ambiguous column name and to ensure there is no ambiguity in the resulting SQL. 
- Validate all conditions against the schema. Ensure value types match column types (e.g., compare DATE columns using DATE(); avoid comparing integers to strings). Rewrite invalid conditions as needed.
- **CRITICAL**: UDF result tables (e.g., `_result`) typically contain ONLY the columns specified in the UDF's Input Columns, NOT all columns from the original tables.
  - To access other columns (like Id, count fields, or any columns not in the UDF's Input Columns), you MUST join the UDF result table back to the original input tables and select from those original tables.

3. Output only the final executable SQL statement.
  - Do not include comments, explanations, or any non-SQL text.
  - The SQL must be valid SQLite syntax and must execute without schema errors.
  - The SQL must end with a ";".


---

## Input:

DB Schema:
{db_schema}

Extended Schema:
{extended_table}

Equivalent relational expression:
{primitive}

Query Execution Plan: 
{json_tree}

----------- 
## Examples
{few_shot}

'''

PLAN2SQL_examples = '''
---------------
DB Schema:
['races.raceId', 'races.name', 'races.year', 'results.raceId', 'results.driverId', 'results.position', 'results.positionText', 'drivers.driverId', 'drivers.forename', 'drivers.surname', 'drivers.dob', 'drivers.nationality']

Extended Schema:

Table: DriversFromF1WorldChampionCountry
created by sql:
CREATE TEMP TABLE DriversFromF1WorldChampionCountry AS
SELECT 
    *
FROM drivers
JOIN results ON drivers."driverId" = results."driverId"
JOIN races ON results."raceId" = races."raceId"
WHERE races."year" = 2009 
  AND races."name" = 'European Grand Prix' 
  AND results."position" <= 3;

Table: ageAtRaceOver25_result_of_DriversFromF1WorldChampionCountry
dob,,TEXT,,,['1972-05-23', '1985-01-07', '1979-10-17']
date,,TEXT,,,['2009-08-23']
ageAtRaceOver25,,TEXT,ageAtRaceOver25 is a predicate derived from the columns ([dob],[date]) in Table DriversFromF1WorldChampionCountry. You can use [dob],[date] to join with Table DriversFromF1WorldChampionCountry,1 means the selection predicate is satisfied and 0 means that the selection predicate is not satisfied,['1', '0']

Equivalent relational expression:

Query Execution Plan: 
{
  "Node Type": "Projection",
  "Columns": ["forename", "surname", "position"],
  "Input": {
    "Node Type": "Selection",
    "UDF": { "UDF Name": "ageAtRaceOver25", "Input Columns": ["dob", "date"] },
    "Input": {
      "Node Type": "Table",
      "Table Name": "DriversFromF1WorldChampionCountry"
      }
  }
}

Output: 
SELECT DriversFromF1WorldChampionCountry."forename", DriversFromF1WorldChampionCountry."surname", DriversFromF1WorldChampionCountry."position"
FROM DriversFromF1WorldChampionCountry
JOIN ageAtRaceOver25_result_of_DriversFromF1WorldChampionCountry ON DriversFromF1WorldChampionCountry."dob" = ageAtRaceOver25_result_of_DriversFromF1WorldChampionCountry."dob"
AND ageAtRaceOver25_result_of_DriversFromF1WorldChampionCountry."date" = DriversFromF1WorldChampionCountry."date"
WHERE ageAtRaceOver25_result_of_DriversFromF1WorldChampionCountry."ageAtRaceOver25" = '1';  
---------------
DB Schema:
['schools.CDSCode', 'schools.County', 'satscores.cds', 'satscores.sname', 'satscores.NumGE1500']

Extended Schema:
Table: SchoolsWithSATScores
created by sql:
CREATE TEMP TABLE SchoolsWithSATScores AS
SELECT
    *
FROM schools
JOIN satscores ON schools."CDSCode" = satscores."cds";

Table: LowestIncome_sort_by_County
County,,TEXT,County name,nan,['Alameda', 'Alpine', 'Amador']
LowestIncome,,int,LowestIncome is extracted from the columns (County) in Table SchoolsWithSATScores. You can use (County) to join this table with SchoolsWithSATScores,a score to indicate how well the item meets the criterion LowestIncome. The score should be between 0 and 100, where higher scores represent better alignment. The score should be a floating-point number.,[1, 2, 3]

Equivalent relational expression:

Query Execution Plan: 
{
  "Node Type": "TopK",
  "k": "inf",
  "UDF": {
    "UDF Name": "LowestIncome",
    "Input Columns": ["County"]
  },
  "Input": {
    "Node Type": "Table",
    "Table Name": "SchoolsWithSATScores"
  }
}

Output:
SELECT *
FROM SchoolsWithSATScores
JOIN LowestIncome_sort_by_County ON SchoolsWithSATScores."County" = LowestIncome_sort_by_County."County"
ORDER BY LowestIncome_sort_by_County."LowestIncome" DESC;
---------------
DB Schema:
['constructors.constructorId', 'constructors.name', 'drivers.driverId', 'drivers.forename', 'drivers.surname', 'races.name', 'races.raceId', 'results.constructorId', 'results.driverId', 'results.raceId']

Extended Schema:
Table: DriverandConstructorsofChineseGrandPrix
created by sql:
CREATE TEMP TABLE DriverandConstructorsofChineseGrandPrix AS
SELECT 
    constructors."name" AS "constructors_name",
    drivers."forename" AS "driver_forename",
    drivers."surname" AS "driver_surname"
FROM drivers
JOIN results ON drivers."driverId" = results."driverId"
JOIN races ON results."raceId" = races."raceId"
JOIN constructors ON results."constructorId" = constructors."constructorId"
WHERE races."name" = 'Chinese Grand Prix';

Table: MostPopularDriver_result_group_by_constructors_name
constructors_name,,TEXT,,,['McLaren', 'Ferrari', 'Renault']
MostPopularDriver,,TEXT,The "MostPopularDriver" column is derived by applying the "MostPopularDriver" operation to the grouped data in Table DriverandConstructorsofChineseGrandPrix based on the columns (['constructors_name']). This aggregation is intended to help answer the question: "For the Chinese Grand Prix, identify the most popular driver from each constructor. Please list the constructor name and the full name of the driver.".,The values in the "MostPopularDriver" column represent the result of the "MostPopularDriver" aggregation operation. Each value summarizes the aggregation outcome for its respective group.,['Lewis Hamilton', 'Felipe Massa', 'Fernando Alonso']

Equivalent relational expression:

Query Execution Plan: 
{
  "Node Type": "Aggregation",
  "UDF": {
    "UDF Name": "MostPopularDriver",
    "Input Columns": ["driver_forename", "driver_surname"],
    "Output Column": "MostPopularDriver"
  },
  "GroupBy Columns": ["constructors_name"],
  "Input": {
    "Node Type": "Table",
    "Table Name": "DriverandConstructorsofChineseGrandPrix"
  }
}

Output:
SELECT 
    MostPopularDriver_result_group_by_constructors_name."constructors_name",
    MostPopularDriver_result_group_by_constructors_name."MostPopularDriver"
FROM MostPopularDriver_result_group_by_constructors_name;
'''

findPrimitve_example = '''
Node Type: "Selection"
UDF Description: 
{{
  "UDF Name": "isInBayArea",
  "Input Columns": ["County"]
}}

List of Values in Input Columns '['County']': 
[{{"Orange"}},{{"San Joaquin"}},{{"Santa Cruz"}},{{"Butte"}}...]
Output:
{{
"result": "isInBayArea is equivalent to County IN ('Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco', 'San Mateo', 'Santa Clara', 'Solano', 'Sonoma')",
"explanation": "The UDF identifies counties located in the San Francisco Bay Area. A precise SQL equivalent is to list all such county names explicitly using an IN clause."
}}
=====
Node Type: "Selection"
UDF Description: 
{{
  "UDF Name": "countyPopulationOver2Million",
  "Input Columns": ["County"]
}}

List of Values in Input Columns '['County']': 
[{{"Napa"}},{{"San Bernardino"}},{{"San Mateo"}},{{"Amador"}}...]
Output:
{{
"result": "countyPopulationOver2Million is equivalent to County IN ('Los Angeles', 'Orange', 'Riverside', 'San Bernardino', 'San Diego', 'Santa Clara')",
"explanation": "This UDF likely identifies counties with populations exceeding 2 million. Based on the sample values and schema, we can infer that the input represents U.S. counties. A suitable SQL equivalent is to explicitly list all such counties using an IN clause."
}}
=====
Node Type: "Selection"
UDF Description: 
{{
  "UDF Name": "isHealingRelated",
  "Input Columns": ["power_name"]
}}

List of Values in Input Columns '['power_name']': 
[{{"Agility"}},{{"Cold Resistance"}},{{"Durability"}},{{"Stealth"}}...]
Output:
{{
"result": "isHealingRelated is equivalent to power_name LIKE '%Healing%'",
"explanation": "This UDF filters powers related to healing. A suitable SQL equivalent checks if the string 'Healing' is present in the power name using a LIKE clause."
}}
=====
Node Type: "Selection"
UDF Description: 
{{
  "UDF Name": "ageAtDateGreaterThan42",
  "Input Columns": ["dob", "date"]
}}

List of Values in Input Columns '['dob', 'date']': 
[{{"1981-07-29","2008-09-28"}},{{"1985-06-27","2008-09-28"}},{{"1985-01-07","2008-09-28"}}...]
Output:
{{
"result": "ageAtDateGreaterThan42 is equivalent to date >= date('dob', '+42 years')",
"explanation": "This UDF checks whether a person is at least 42 years old on a given date. This can be computed by adding 42 years to the date of birth and comparing it to the reference date."
}}
=====
Node Type: "Selection"
UDF Description: 
{{
  "UDF Name": "isNorthOfWashingtonMonument",
  "Input Columns": ["Latitude"]
}}

List of Values in Input Columns '['Latitude']': 
[{{"37.521436"}},{{"37.782147"}},{{"37.779884"}},{{"37.764294"}}...]
Output:
{{
"result": "isNorthOfWashingtonMonument is equivalent to Latitude > 38.8895",
"explanation": "This UDF checks whether a location is north of the Washington Monument. Since the monument’s latitude is approximately 38.8895, a simple comparison using Latitude > 38.8895 suffices."
}}
=====
Node Type: "Projection"
UDF Description: 
{{
  "UDF Name": "calculateDriverAge",
  "Input Columns": ["dob"],
  "Output Column": "age"
}}

List of Values in Input Columns '['Latitude']': 
[{{"1985-01-07"}},{{"1977-05-10"}},{{"1985-06-27"}}...]
Output:
{{
"result": "calculateDriverAge is equivalent to (strftime('%Y', 'now') - strftime('%Y', dob)) - (strftime('%m-%d', 'now') < strftime('%m-%d', dob)) ",
"explanation": "This UDF calculates a driver's current age based on their date of birth. The expression subtracts the birth year from the current year and adjusts by checking if the birthday has already occurred this year."
}}
=====
'''
