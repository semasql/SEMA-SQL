#!/bin/bash
# Evaluation script for SEMA-SQL
# This script runs a single query evaluation using the SEMA-SQL framework

python eval/single_evaluation.py \
    --db_path "data/MINIDEV/dev_databases" \
    --db_name california_schools \
    --query "Among the schools with the average score in Math over 560 in the SAT test, how many schools are in the bay area?" \
    --llm_model claude \
    --column_selector 1 \
    --question_decomposition 1 \
    --optimization_lazy_llm 1 \
    --optimization_udf_rewrite 1 \
    ${@}
