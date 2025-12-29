#!/usr/bin/env python3
"""
Script to update all YAML config files with GNA settings.
Handles both main simulation section and nested eval_checkpoint.simulation_config.
"""

import os
import re
import glob

def add_gna_after_use_multi(content):
    """Add GNA settings after ALL use_multi lines that don't already have GNA."""
    
    lines = content.split('\n')
    new_lines = []
    i = 0
    changes_made = 0
    
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # Check if this is a use_multi line
        if re.match(r'^(\s*)use_multi:\s*(true|false)\s*$', line):
            indent = re.match(r'^(\s*)', line).group(1)
            
            # Check if next line already has enable_gna
            if i + 1 < len(lines) and 'enable_gna' in lines[i + 1]:
                # Already has GNA
                pass
            else:
                # Add GNA settings with same indent
                new_lines.append(f'{indent}enable_gna: true')
                new_lines.append(f'{indent}gna_top_k: 3')
                new_lines.append(f'{indent}gna_selection_mode: priority  # "priority" or "random"')
                changes_made += 1
        
        i += 1
    
    return '\n'.join(new_lines), changes_made

def add_fov_limits(content):
    """Add max_local_entities and max_global_entities to ALL fov_entities sections."""
    
    lines = content.split('\n')
    new_lines = []
    i = 0
    changes_made = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a fov_entities line
        fov_match = re.match(r'^(\s*)fov_entities:\s*$', line)
        if fov_match:
            fov_indent = fov_match.group(1)
            new_lines.append(line)
            i += 1
            
            # Next line should be Entity: N
            if i < len(lines):
                entity_match = re.match(r'^(\s*)Entity:\s*(\d+)\s*(#.*)?$', lines[i])
                if entity_match:
                    entity_indent = entity_match.group(1)
                    entity_count = int(entity_match.group(2))
                    
                    # Check if next line already has max_local_entities
                    if i + 1 < len(lines) and 'max_local_entities' in lines[i + 1]:
                        # Already has limits, keep as is
                        new_lines.append(lines[i])
                    else:
                        # Calculate split
                        max_global = min(2, entity_count - 1)
                        max_local = entity_count - max_global
                        
                        # Add updated Entity line with limits
                        new_lines.append(f'{entity_indent}Entity: {entity_count}                        # Maximum number of agents total (local + global)')
                        new_lines.append(f'{entity_indent}max_local_entities: {max_local}            # Maximum number of entities from local FOV')
                        new_lines.append(f'{entity_indent}max_global_entities: {max_global}           # Maximum number of entities from GNA broadcast')
                        changes_made += 1
                    i += 1
                    continue
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines), changes_made

def clean_duplicate_comments(content):
    """Remove duplicate comment lines that may have been created."""
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        # Skip lines that are just old comments about Entity
        if re.match(r'^#\s*Maximum number of agents', line.strip()):
            continue
        if re.match(r'^\s*#\s*Maximum number of agents', line) and i > 0:
            # Check if previous line already has this comment
            if 'Maximum number of agents' in new_lines[-1]:
                continue
        new_lines.append(line)
    
    return '\n'.join(new_lines)

def process_file(filepath):
    """Process a single YAML file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Add GNA settings after all use_multi lines
        content, gna_count = add_gna_after_use_multi(content)
        
        # Add fov_entities limits
        content, fov_count = add_fov_limits(content)
        
        # Clean up any duplicate comments
        content = clean_duplicate_comments(content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True, gna_count, fov_count
        
        return False, 0, 0
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False, 0, 0

def main():
    base_dir = "/Users/ahmetfaruksaz/GitHub/CityLogi/LogiCity/config/tasks"
    
    # Find all YAML files
    yaml_files = glob.glob(os.path.join(base_dir, "**/*.yaml"), recursive=True)
    
    print(f"Found {len(yaml_files)} YAML files")
    print("=" * 60)
    
    updated_count = 0
    total_gna_added = 0
    total_fov_added = 0
    
    for filepath in sorted(yaml_files):
        updated, gna_count, fov_count = process_file(filepath)
        
        if updated:
            updated_count += 1
            total_gna_added += gna_count
            total_fov_added += fov_count
            rel_path = os.path.relpath(filepath, base_dir)
            changes = []
            if gna_count > 0:
                changes.append(f"GNA×{gna_count}")
            if fov_count > 0:
                changes.append(f"FOV×{fov_count}")
            print(f"Updated: {rel_path} [{', '.join(changes)}]")
    
    print("=" * 60)
    print(f"Total files updated: {updated_count}")
    print(f"  - GNA sections added: {total_gna_added}")
    print(f"  - FOV limit sections added: {total_fov_added}")

if __name__ == "__main__":
    main()

