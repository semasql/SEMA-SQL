"""
Single query evaluation script for SEMA-SQL.

This script evaluates a single natural language query against a database using
LLM-enhanced query execution. It demonstrates the complete pipeline:
1. Schema filtering
2. Question decomposition
3. Query generation (NL2JSON)
4. Query optimization
5. Query execution
"""

import argparse
import asyncio
import json
import os
import sys
import warnings
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.semasql.core.llm_query_executor import LLMEnhancedDBExecutor

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate a single natural language query using SEMA-SQL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--db_path',
        type=str,
        default='data/MINIDEV/dev_databases',
        help='Path to the directory containing database files'
    )
    parser.add_argument(
        '--db_name',
        type=str,
        default='california_schools',
        help='Name of the database to query'
    )
    parser.add_argument(
        '--query',
        type=str,
        default='Among the schools with the average score in Math over 560 in the SAT test, how many schools are in the bay area?',
        help='Natural language query to execute'
    )
    parser.add_argument(
        '--llm_model',
        type=str,
        default='gpt',
        choices=['gpt', 'claude', 'qwen', 'gemini'],
        help='LLM model to use for query processing'
    )
    parser.add_argument(
        '--column_selector',
        type=int,
        default=1,
        choices=[0, 1],
        help='Enable schema filtering (1) or disable (0)'
    )
    parser.add_argument(
        '--question_decomposition',
        type=int,
        default=1,
        choices=[0, 1],
        help='Enable question decomposition (1) or disable (0)'
    )
    parser.add_argument(
        '--optimization_lazy_llm',
        type=int,
        default=1,
        choices=[0, 1],
        help='Enable UDF deferral optimization (1) or disable (0)'
    )
    parser.add_argument(
        '--optimization_udf_rewrite',
        type=int,
        default=1,
        choices=[0, 1],
        help='Enable UDF rewriting optimization (1) or disable (0)'
    )
    
    return parser.parse_args()


def check_and_clean_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Ensure a clean event loop for async operations.
    
    Closes any existing event loop and creates a new one to avoid
    conflicts with existing async contexts.
    
    Returns:
        asyncio.AbstractEventLoop: New event loop instance.
    """
    try:
        current_loop = asyncio.get_event_loop()
        if not current_loop.is_closed():
            current_loop.close()
    except RuntimeError:
        pass
    
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    return new_loop


def print_token_usage(phase: str, total: int, input_tokens: int, output_tokens: int) -> None:
    """Print token usage statistics for a given phase.
    
    Args:
        phase: Name of the execution phase
        total: Total tokens used
        input_tokens: Input tokens used
        output_tokens: Output tokens used
    """
    print(f"\n[{phase}] Token Usage: Total={total}, Input={input_tokens}, Output={output_tokens}")


async def main() -> None:
    """Main execution function for single query evaluation."""
    args = parse_args()
    check_and_clean_event_loop()
    
    print("=" * 60)
    print("SEMA-SQL Single Query Evaluation")
    print("=" * 60)
    print(f"Database: {args.db_name}")
    print(f"Query: {args.query}")
    print(f"Model: {args.llm_model}")
    print("=" * 60)
    
    try:
        # Initialize the query executor
        query_executor = LLMEnhancedDBExecutor(
            path=args.db_path,
            db_name=args.db_name,
            question=args.query,
            model=args.llm_model,
            enable_column_selector=args.column_selector,
            enable_question_decomposition=args.question_decomposition,
            optimization_lazy_llm=args.optimization_lazy_llm,
            optimization_udf_rewrite=args.optimization_udf_rewrite
        )
        
        # Phase 0: Schema filtering
        print("\n[Phase 0] Schema Filtering")
        print("-" * 60)
        query_executor.filter_schema()
        print("Filtered schema:")
        print(query_executor.schema)
        token_usage_p0, input_tokens_p0, output_tokens_p0 = query_executor.llm_model.get_total_tokens_used()
        print_token_usage("Phase 0", token_usage_p0, input_tokens_p0, output_tokens_p0)
        
        # Phase 1: Question decomposition and plan generation
        print("\n[Phase 1] Question Decomposition & Plan Generation")
        print("-" * 60)
        query_executor.question_decomposition()
        print("Question analysis:")
        print(query_executor.Hint)
        
        query_plan = query_executor.NL2JSON()
        print("\nGenerated query plan (JSON):")
        print(json.dumps(query_plan, indent=2, ensure_ascii=False))
        
        total_tokens_p1, input_tokens_p1, output_tokens_p1 = query_executor.llm_model.get_total_tokens_used()
        print_token_usage("Phase 1", total_tokens_p1, input_tokens_p1, output_tokens_p1)
        
        # Phase 2: Query optimization
        print("\n[Phase 2] Query Optimization")
        print("-" * 60)
        query_executor.query_optimization()
        optimized_plan = json.dumps(query_executor.json_tree, indent=2, ensure_ascii=False)
        print("Optimized query plan:")
        print(optimized_plan)
        
        total_tokens_p2, input_tokens_p2, output_tokens_p2 = query_executor.llm_model.get_total_tokens_used()
        print_token_usage("Phase 2", total_tokens_p2, input_tokens_p2, output_tokens_p2)
        
        # Phase 3: Query execution
        print("\n[Phase 3] Query Execution")
        print("-" * 60)
        final_result = await query_executor.execute_json()
        print("Final result:")
        print(final_result)
        
        total_tokens_p3, input_tokens_p3, output_tokens_p3 = query_executor.llm_model.get_total_tokens_used()
        print_token_usage("Phase 3", total_tokens_p3, input_tokens_p3, output_tokens_p3)
        
        # Summary
        print("\n" + "=" * 60)
        print("Evaluation Complete")
        print("=" * 60)
        print_token_usage("Total", total_tokens_p3, input_tokens_p3, output_tokens_p3)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())