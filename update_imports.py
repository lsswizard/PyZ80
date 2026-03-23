#!/usr/bin/env python3
"""
Script to update import statements from ..cpu to ..core in the core/ folder.
"""

import os
import re

def update_file(filepath):
    """Update a single file, replacing ..cpu with ..core in import statements."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace import statements: from ..cpu import ... to from ..core import ...
        updated_content = re.sub(r'from\s+\.\.cpu\s+', 'from ..core ', content)

        # Check if any changes were made
        if updated_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"Updated: {filepath}")
            return True
        else:
            print(f"No changes needed: {filepath}")
            return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function to process all Python files in the core/ directory."""
    core_dir = 'core'

    if not os.path.exists(core_dir):
        print(f"Directory {core_dir} not found!")
        return

    updated_files = 0
    total_files = 0

    # Walk through all files in the core directory
    for root, dirs, files in os.walk(core_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                total_files += 1
                if update_file(filepath):
                    updated_files += 1

    print(f"\nProcessing complete!")
    print(f"Total Python files processed: {total_files}")
    print(f"Files updated: {updated_files}")

if __name__ == "__main__":
    main()