"""
Find which .circles file contains a specific circle/group.
"""

import os
import sys

def find_circle_source(circle_name):
    """Find which ego network's .circles file contains the given circle."""
    raw_dir = 'data/raw/facebook/facebook'
    
    if not os.path.exists(raw_dir):
        print(f"Raw data directory not found: {raw_dir}")
        return
    
    circle_files = [f for f in os.listdir(raw_dir) if f.endswith('.circles')]
    
    print("=" * 60)
    print(f"SEARCHING FOR CIRCLE: {circle_name}")
    print("=" * 60)
    print(f"\nSearching in {len(circle_files)} circle files...\n")
    
    found_in = []
    
    for circle_file in sorted(circle_files):
        file_path = os.path.join(raw_dir, circle_file)
        ego_id = circle_file.replace('.circles', '')
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) > 0 and parts[0] == circle_name:
                        # Found it!
                        members = parts[1:] if len(parts) > 1 else []
                        found_in.append({
                            'file': circle_file,
                            'ego_id': ego_id,
                            'members': [int(m) for m in members if m.isdigit()],
                            'num_members': len(members)
                        })
                        break
        except Exception as e:
            print(f"Error reading {circle_file}: {e}")
    
    if found_in:
        print(f"[FOUND] {circle_name} appears in {len(found_in)} file(s):\n")
        for item in found_in:
            print(f"File: {item['file']}")
            print(f"Ego Network ID: {item['ego_id']}")
            print(f"Number of members: {item['num_members']}")
            print(f"Members (first 20): {item['members'][:20]}")
            if len(item['members']) > 20:
                print(f"... and {len(item['members']) - 20} more")
            
            # Check if users 828 and 798 are in this circle
            if 828 in item['members']:
                print(f"  [YES] User 828 IS a member")
            else:
                print(f"  [NO] User 828 is NOT a member")
            
            if 798 in item['members']:
                print(f"  [YES] User 798 IS a member")
            else:
                print(f"  [NO] User 798 is NOT a member")
            
            print()
    else:
        print(f"[NOT FOUND] {circle_name} not found in any .circles file")
        print("\nNote: The circle name might be different in the processed data.")
        print("Checking what circles exist in each file...")
        
        # Show what circles exist
        print("\nCircles found in each file:")
        for circle_file in sorted(circle_files)[:10]:  # Show first 10
            file_path = os.path.join(raw_dir, circle_file)
            ego_id = circle_file.replace('.circles', '')
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    circle_names = [line.strip().split('\t')[0] for line in lines if line.strip()]
                    if circle_names:
                        print(f"  {circle_file}: {', '.join(circle_names[:5])}")
                        if len(circle_names) > 5:
                            print(f"    ... and {len(circle_names) - 5} more")
            except:
                pass
    
    print("=" * 60)

if __name__ == '__main__':
    circle_name = sys.argv[1] if len(sys.argv) > 1 else 'circle13'
    find_circle_source(circle_name)

