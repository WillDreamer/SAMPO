import json
import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Any


class OutputChecker:
    """Extensible output checker for analyzing alfworld task output format"""
    
    def __init__(self):
        self.checks = [
            self.check_tag_pairing,
            self.check_strict_format,
            self.check_chinese_characters,
            self.check_repeated_output,
        ]
        
        self.stats = defaultdict(int)
        self.total_samples = 0
        
        # Failure cases storage
        self.failure_cases = defaultdict(list)
        
    # def check_basic_tags(self, output: str) -> Dict[str, Any]:
    #     """Check for the presence of basic think and action tags"""
    #     has_think_open = bool(re.search(r'<\s*think\s*>', output, re.IGNORECASE))
    #     has_action_open = bool(re.search(r'<\s*action\s*>', output, re.IGNORECASE))
        
    #     result = {
    #         'name': 'basic_tags',
    #         'has_think_open': has_think_open,
    #         'has_action_open': has_action_open,
    #         'has_both': has_think_open and has_action_open,
    #         'passed': has_think_open and has_action_open,
    #     }
        
    #     if not has_think_open:
    #         result['issue'] = 'missing_think_open_tag'
    #     elif not has_action_open:
    #         result['issue'] = 'missing_action_open_tag'
            
    #     return result
    
    def check_tag_pairing(self, output: str) -> Dict[str, Any]:
        """Check if tags appear in pairs"""
        think_open = len(re.findall(r'<\s*think\s*>', output, re.IGNORECASE))
        think_close = len(re.findall(r'</\s*think\s*>', output, re.IGNORECASE))
        action_open = len(re.findall(r'<\s*action\s*>', output, re.IGNORECASE))
        action_close = len(re.findall(r'</\s*action\s*>', output, re.IGNORECASE))
        
        result = {
            'name': 'tag_pairing',
            'think_open_count': think_open,
            'think_close_count': think_close,
            'action_open_count': action_open,
            'action_close_count': action_close,
            'think_paired': think_open == think_close == 1,
            'action_paired': action_open == action_close == 1,
            'passed': (think_open == think_close == 1) and (action_open == action_close == 1),
        }
        
        if think_open == 0:
            result['issue'] = 'no_think_open'
        elif think_open > think_close:
            result['issue'] = 'unclosed_think_tag'
        elif think_open < think_close:
            result['issue'] = 'extra_think_close_tag'
        elif action_open > action_close:
            result['issue'] = 'unclosed_action_tag'
        elif action_open < action_close:
            result['issue'] = 'extra_action_close_tag'
        elif think_open > 1 or action_open > 1:
            result['issue'] = 'multiple_tag_pairs'
            
        return result
    
    def check_strict_format(self, output: str) -> Dict[str, Any]:
        """Check if the output strictly follows the <think>...</think><action>...</action> format"""
        strict_pattern = re.compile(
            r'^\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*$',
            flags=re.IGNORECASE | re.DOTALL
        )
        match = strict_pattern.match(output)
        
        result = {
            'name': 'strict_format',
            'passed': bool(match),
        }
        
        if match:
            think_content = match.group(1).strip()
            action_content = match.group(2).strip()
            result['think_length'] = len(think_content)
            result['action_length'] = len(action_content)
        else:
            result['issue'] = 'format_not_strict'
            
        return result
    
    # def check_think_too_long(self, output: str) -> Dict[str, Any]:
    #     """Check if the think tag is too long causing no action output"""
    #     has_think_open = bool(re.search(r'<\s*think\s*>', output, re.IGNORECASE))
    #     has_action_open = bool(re.search(r'<\s*action\s*>', output, re.IGNORECASE))
        
    #     # Extract think content (if exists)
    #     think_match = re.search(r'<think>(.*?)(?:</think>|$)', output, re.IGNORECASE | re.DOTALL)
    #     think_length = len(think_match.group(1)) if think_match else 0
        
    #     # Define threshold (adjustable)
    #     LONG_THINK_THRESHOLD = 1500
        
    #     result = {
    #         'name': 'think_too_long',
    #         'think_length': think_length,
    #         'is_long': think_length > LONG_THINK_THRESHOLD,
    #         'missing_action': has_think_open and not has_action_open,
    #         'passed': not (has_think_open and not has_action_open and think_length > LONG_THINK_THRESHOLD),
    #     }
        
    #     if has_think_open and not has_action_open and think_length > LONG_THINK_THRESHOLD:
    #         result['issue'] = 'think_too_long_no_action'
    #     elif has_think_open and not has_action_open:
    #         result['issue'] = 'has_think_no_action'
            
    #     return result
    
    # def check_incomplete_output(self, output: str) -> Dict[str, Any]:
    #     """Check if the output is incomplete (truncated)"""
    #     # Check if it ends in an incomplete way
    #     incomplete_patterns = [
    #         r'<think>.*(?!</think>)$',  # think tag not closed
    #         r'</think>\s*$',  # only think, no action
    #         r'<action>[^<]*$',  # action tag content not completed
    #     ]
        
    #     is_incomplete = False
    #     incomplete_type = None
        
    #     for i, pattern in enumerate(incomplete_patterns):
    #         if re.search(pattern, output, re.IGNORECASE | re.DOTALL):
    #             is_incomplete = True
    #             if i == 0:  # think tag not closed
    #                 incomplete_type = 'unclosed_think'
    #             elif i == 1:  # only think, no action
    #                 incomplete_type = 'missing_action_after_think'
    #             elif i == 2:  # action tag content not completed
    #                 incomplete_type = 'incomplete_action'
    #             break
        
    #     result = {
    #         'name': 'incomplete_output',
    #         'is_incomplete': is_incomplete,
    #         'passed': not is_incomplete,
    #     }
        
    #     if is_incomplete:
    #         result['issue'] = incomplete_type
            
    #     return result
    
    def check_chinese_characters(self, output: str) -> Dict[str, Any]:
        """Check for presence of Chinese characters in the output"""
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', output))
        
        result = {
            'name': 'chinese_characters',
            'has_chinese': has_chinese,
            'passed': not has_chinese,
        }
        
        if has_chinese:
            result['issue'] = 'contains_chinese'

            chinese_matches = list(re.finditer(r'[\u4e00-\u9fff]+', output))
            result['chinese_count'] = len(chinese_matches)
            result['chinese_samples'] = [m.group() for m in chinese_matches[:3]]  # Keep only first 3 samples
            
        return result

    def check_repeated_output(self, output: str) -> Dict[str, Any]:
        """Detect the same word repeated consecutively (default: >=5 times)"""
        pattern = re.compile(r"\b([a-zA-Z0-9]+)(?:\s+\1){4,}\b", flags=re.IGNORECASE)
        repetitions: List[Dict[str, Any]] = []

        for match in pattern.finditer(output):
            word = match.group(1)
            segment = match.group(0)
            count = len(re.findall(rf"\b{re.escape(word)}\b", segment, flags=re.IGNORECASE))
            repetitions.append({
                'word': word,
                'count': count,
                'excerpt': segment,
            })

        result = {
            'name': 'repeated_output',
            'has_repetition': bool(repetitions),
            'repetitions': repetitions,
            'passed': not repetitions,
        }

        if repetitions:
            result['issue'] = 'repeated_output'

        return result
    
    # def check_tag_order(self, output: str) -> Dict[str, Any]:
    #     """Check if the tag order is correct (think should be before action)"""
    #     think_match = re.search(r'<\s*think\s*>', output, re.IGNORECASE)
    #     action_match = re.search(r'<\s*action\s*>', output, re.IGNORECASE)
        
    #     correct_order = True
    #     if think_match and action_match:
    #         correct_order = think_match.start() < action_match.start()
        
    #     result = {
    #         'name': 'tag_order',
    #         'has_both_tags': bool(think_match and action_match),
    #         'correct_order': correct_order,
    #         'passed': correct_order if (think_match and action_match) else True,
    #     }
        
    #     if think_match and action_match and not correct_order:
    #         result['issue'] = 'action_before_think'
            
    #     return result
    
    def analyze_output(self, output: str, input: str, idx: int = None) -> Dict[str, Any]:
        """Perform all checks on a single output"""
        self.total_samples += 1
        
        results = {
            'output_length': len(output),
            'checks': {},
            'all_passed': True,
            'issues': [],
        }
        
        for check_func in self.checks:
            check_result = check_func(output)
            check_name = check_result['name']
            results['checks'][check_name] = check_result
            
            # Update statistics
            if check_result.get('passed'):
                self.stats[f'{check_name}_passed'] += 1
            else:
                self.stats[f'{check_name}_failed'] += 1
                results['all_passed'] = False
                
                if 'issue' in check_result:
                    issue = check_result['issue']
                    results['issues'].append(issue)
                    self.stats[f'issue_{issue}'] += 1
                    
                    # Collect failure cases (limit quantity)
                    if len(self.failure_cases[issue]) < 5:
                        self.failure_cases[issue].append({
                            'index': idx,
                            'output_preview': output,
                            'check_result': check_result,
                            "input": input,
                        })
        
        return results
    
    def print_statistics(self):
        """Print statistics results"""
        print("\n" + "="*80)
        print(f"Total Samples: {self.total_samples}")
        print("="*80)
        
        # Display grouped by check category
        print("\n[Check Pass Rate]")
        check_names = set()
        for key in self.stats.keys():
            if key.endswith('_passed') or key.endswith('_failed'):
                check_name = key.rsplit('_', 1)[0]
                check_names.add(check_name)
        
        for check_name in sorted(check_names):
            passed = self.stats.get(f'{check_name}_passed', 0)
            failed = self.stats.get(f'{check_name}_failed', 0)
            total = passed + failed
            if total > 0:
                pass_rate = passed / total * 100
                print(f"  {check_name:25s}: {passed:4d}/{total:4d} ({pass_rate:5.1f}%)")
        
        # Display common issues
        print("\n[Common Issues Statistics]")
        issue_stats = [(k.replace('issue_', ''), v) for k, v in self.stats.items() if k.startswith('issue_')]
        issue_stats.sort(key=lambda x: x[1], reverse=True)
        
        for issue, count in issue_stats:
            percentage = count / self.total_samples * 100
            print(f"  {issue:35s}: {count:4d} ({percentage:5.1f}%)")
        
    def print_failure_cases(self, max_issues: int = 5):
        """Print failure case examples"""
        print("\n" + "="*80)
        print("[Failure Case Examples]")
        print("="*80)
        
        for issue, cases in list(self.failure_cases.items())[:max_issues]:
            if issue == 'contains_chinese':
                continue  # Skip this issue for failure case display
            print(f"\nIssue Type: {issue}")
            print("-" * 80)
            for i, case in enumerate(cases, 1):
                print(f"\n  Case {i} (Index: {case.get('index', 'N/A')}):")
                print(f"  Output Preview: {case['output_preview']}")
                print(f"  Input: {case.get('input', 'N/A')}")
                check_result = case['check_result']
                for key, value in check_result.items():
                    if key not in ['name', 'passed', 'issue']:
                        print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description='Check alfworld output file format and quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Check single step output
  python check_output.py --file_path /path/to/outputs --start_step 1 --end_step 1
  
  # Check multiple steps output
  python check_output.py --file_path /path/to/outputs --start_step 1 --end_step 10
  
  # Show only statistics, no failure cases
  python check_output.py --file_path /path/to/outputs --start_step 1 --end_step 10 --no-cases
  
  # Show detailed analysis for each sample
  python check_output.py --file_path /path/to/outputs --start_step 1 --end_step 1 --verbose
        """
    )
    parser.add_argument("--file_path", type=str, required=True, help="Output file directory")
    parser.add_argument("--start_step", type=int, required=True, help="Start step number")
    parser.add_argument("--end_step", type=int, required=True, help="End step number")
    parser.add_argument("--verbose", action="store_true", help="Show detailed analysis for each sample")
    parser.add_argument("--no-cases", action="store_true", help="Do not show failure case examples")
    parser.add_argument("--max-issues", type=int, default=5, help="Maximum number of issue types to display")
    
    args = parser.parse_args()
    
    checker = OutputChecker()
    
    # Read and analyze all files
    for step in range(args.start_step, args.end_step + 1):
        file_path = os.path.join(args.file_path, f"{step}.jsonl")
        
        if not os.path.exists(file_path):
            print(f"Warning: File does not exist - {file_path}")
            continue
        
        print(f"Processing: {file_path}")
        
        with open(file_path, "r") as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    output = data.get("output", "")
                    input = data.get("input", "")
                    if not output:
                        continue
                    
                    result = checker.analyze_output(output, input=input, idx=(step, idx))
                    
                    if args.verbose:
                        print(f"\nSample {step}-{idx}:")
                        print(f"  Length: {result['output_length']}")
                        print(f"  Passed: {result['all_passed']}")
                        if result['issues']:
                            print(f"  Issues: {', '.join(result['issues'])}")
                        
                except json.JSONDecodeError as e:
                    print(f"Error: Cannot parse JSON - Step {step}, Line {idx}: {e}")
                except Exception as e:
                    print(f"Error: Step {step}, Line {idx}: {e}")
    
    # Print statistics results
    checker.print_statistics()
    
    # Print failure cases
    if not args.no_cases:
        checker.print_failure_cases(max_issues=args.max_issues)


if __name__ == "__main__":
    main()
